"""E8 の実行を、同時に N 本ずつ続けて回す。

v1 README の「主張と実験の記録」の E8 を、実行待ちの一覧にして順に回す。
1本終わるたびに次を始め、失敗した実行は記録して次へ進む。
学習そのものは今までどおり `proposed_main.py` と `main.py` が行い、このファイルは呼び出すだけである。

使い方（このファイルのあるディレクトリで実行する）:

    ../../.venvs/mujoco/bin/python run_queue.py plan                    # 実行待ちの一覧を作り、本数を表示する
    ../../.venvs/mujoco/bin/python run_queue.py run --workers 2 --cuda 0  # 同時に2本ずつ、終わるまで回す
    ../../.venvs/mujoco/bin/python run_queue.py run --env Reacher-v4     # Reacher だけ回す
    ../../.venvs/mujoco/bin/python run_queue.py status                  # 終わった本数・残り・失敗を表示する

研究室サーバ3台で分けて回すときは、SERVER_TASKS の表に従って `--server` で自分の分だけを回す。
各サーバでは screen の中で、その pyenv の仮想環境の Python を使う:

    python run_queue.py plan
    python run_queue.py run --server 1 --workers 2      # 2番サーバでは --server 2、3番サーバでは --server 3
    python run_queue.py status --server 1

止めたあと同じコマンドで再開すると、終わった実行は飛ばす。
終わったかどうかは各サーバのディスクで判定するので、1つのタスクは1台だけで回す。

出力:

    models/e8_{タスク}/...            各実行の結果。main.py / proposed_main.py が書く
    models/_queue/e8_jobs.csv          実行待ちの一覧（plan が書く）
    models/_queue/e8_runs.csv          1本終わるごとに1行。開始・終了・秒数・終了コード（run が書く）
    models/_queue/logs/{実行ID}.log    各実行の画面出力とエラー

終わったかどうかは、保存先に eval_metrics*.csv が揃っているかで判定する。
この CSV は学習が最後まで終わったときだけ書かれるので、途中で止まった実行はもう一度回す。
秒数は同時に回す本数で変わるので、手法の比較ではなく残り時間の目安にだけ使う。
"""

import argparse
import csv
import datetime
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE_DIR = os.path.join('models', '_queue')
JOBS_CSV = os.path.join(QUEUE_DIR, 'e8_jobs.csv')
RUNS_CSV = os.path.join(QUEUE_DIR, 'e8_runs.csv')
LOG_DIR = os.path.join(QUEUE_DIR, 'logs')

# ----------------------------------------------------------------------
# E8 の定義。変えたら v1 README の E8 にも日付と理由を書き足す
# ----------------------------------------------------------------------

# 環境 ID → (config の名前, 1体あたりの学習ステップ数)。並びは1本の時間が長い順
TASKS = {
    'Humanoid-v4': ('humanoid', 10_000_000),
    'Walker2d-v4': ('walker2d', 10_000_000),
    'Hopper-v4': ('hopper', 3_000_000),
    'HalfCheetah-v4': ('halfcheetah', 3_000_000),
    'Reacher-v4': ('reacher', 1_000_000),
    'Pusher-v4': ('pusher', 1_000_000),
}

SEEDS = [2021, 2022, 2023, 2024, 2025]

# 提案手法の3設定：中間発表の 153 設定の Reacher スコア上位3つ
SETTINGS = [
    # (番号, α ペア, ASI, GSI, τ)
    (1, '0.01,0.5', 7000, 1000, 0.3),
    (2, '0.001,0.1', 9000, 3000, 0.3),
    (3, '0.01,0.1', 9000, 1000, 0.3),
]

# 学習シード 2021 の実行が plot/ に既にあり、回し直さない組
EXISTING_PROPOSED_2021 = {
    (1, 'Reacher-v4'), (1, 'Hopper-v4'), (1, 'Walker2d-v4'), (1, 'Humanoid-v4'),
    (2, 'Reacher-v4'), (3, 'Reacher-v4'),
}
EXISTING_SAC_2021 = {'Reacher-v4', 'Hopper-v4', 'Walker2d-v4', 'Humanoid-v4'}

# SAC-v1 のタスクごとの最良 α。既存の学習シード 2021 の実行で、上のステップ数の時点に最も良かった値。
# HalfCheetah と Pusher は、SWEEP_ALPHAS を 2021 で回して決めたらここに書く。None の間は 2022〜2025 を一覧に入れない
SAC_V1_ALPHA = {
    'Reacher-v4': 0.022,
    'Hopper-v4': 0.05,
    'Walker2d-v4': 0.01,
    'Humanoid-v4': 0.2,
    'HalfCheetah-v4': None,
    'Pusher-v4': None,
}
SWEEP_ALPHAS = [0.0001, 0.01, 0.05, 0.1, 0.2]

# 研究室サーバ3台への振り分け。サーバどうしはつながっていないので、タスクごとに1台へ割り当てる。
# 3台は同じ設定・同じシードで結果が一致することを確認済み（ユーザ、2026-10-18）。
# 変えるときは、まだ始めていないタスクだけを動かす。始めたタスクを移すと、移した先で同じ実行をもう一度回してしまう
SERVER_TASKS = {
    1: ['Humanoid-v4'],
    2: ['Walker2d-v4'],
    3: ['Hopper-v4', 'HalfCheetah-v4', 'Reacher-v4', 'Pusher-v4'],
}

JOB_FIELDS = ['job_id', 'phase', 'method', 'env', 'seed', 'setting', 'alpha', 'asi', 'gsi', 'tau_g',
              'num_steps', 'config', 'exp_id', 'save_dir', 'cmd']
RUN_FIELDS = ['job_id', 'method', 'env', 'seed', 'server', 'host', 'start', 'end', 'seconds', 'returncode', 'status',
              'cuda', 'save_dir', 'log']


# ----------------------------------------------------------------------
# 実行待ちの一覧を作る
# ----------------------------------------------------------------------

def proposed_job(setting, env, seed):
    no, alpha, asi, gsi, tau = setting
    name, steps = TASKS[env]
    exp_id = f'e8_{name}'
    # proposed_main.py の save_path と同じ組み立て方【コード】proposed_main.py の save_path
    alpha_list = str([float(a) for a in alpha.split(',')])
    save_dir = f'models/{exp_id}/{alpha_list}/v1/{seed}/True/tau_g{tau}_gsi{gsi}_asi{asi}/'
    args = ['proposed_main.py', f'configs_pro/{name}1.yml', '--seed', str(seed), '--alpha', alpha,
            '--gtoq', str(asi), '--qtog', str(gsi), '--tau_g', str(tau), '--num_steps', str(steps),
            '--exp_id', exp_id]
    return dict(job_id=f'pro{no}-{name}-{seed}', phase='main', method='proposed', env=env, seed=seed,
                setting=no, alpha=alpha, asi=asi, gsi=gsi, tau_g=tau, num_steps=steps,
                config=args[1], exp_id=exp_id, save_dir=save_dir, cmd=' '.join(args))


def sac_job(version, env, seed, alpha=None, phase='main'):
    name, steps = TASKS[env]
    exp_id = f'e8_{name}'
    config = f'configs/{name}{1 if version == "v1" else 2}.yml'
    args = ['main.py', config, '--seed', str(seed), '--num_steps', str(steps), '--exp_id', exp_id]
    if alpha is not None:
        args += ['--alpha', str(alpha)]
        alpha_dir = str(float(alpha))
    else:
        alpha_dir = '0.2'  # configs/*2.yml の alpha。SAC-v2 では初期値で、学習中に自動で調整される
    save_dir = f'models/{exp_id}/{alpha_dir}/{version}/{seed}/'
    tag = f'sacv1a{alpha}' if version == 'v1' else 'sacv2'
    return dict(job_id=f'{tag}-{name}-{seed}', phase=phase, method=f'sac-{version}', env=env, seed=seed,
                setting='', alpha=alpha if alpha is not None else '', asi='', gsi='', tau_g='',
                num_steps=steps, config=config, exp_id=exp_id, save_dir=save_dir, cmd=' '.join(args))


def build_jobs():
    sweep, main = [], []
    for env in TASKS:
        # SAC-v1 の α 選び。結果が出ないと残りのシードを決められないので、一覧の先頭に置く
        if SAC_V1_ALPHA[env] is None:
            sweep += [sac_job('v1', env, 2021, a, phase='alpha-sweep') for a in SWEEP_ALPHAS]
        for seed in SEEDS:
            for s in SETTINGS:
                if seed == 2021 and (s[0], env) in EXISTING_PROPOSED_2021:
                    continue
                main.append(proposed_job(s, env, seed))
            if not (seed == 2021 and env in EXISTING_SAC_2021):
                main.append(sac_job('v2', env, seed))
                # α 選びのタスクでは、2021 の最良 α は α 選びの実行そのものなので足さない
                if SAC_V1_ALPHA[env] is not None and not (seed == 2021 and env not in EXISTING_SAC_2021):
                    main.append(sac_job('v1', env, seed, SAC_V1_ALPHA[env]))
    return sweep + main


def check_servers():
    """どのタスクもちょうど1台に割り当てられているかを確かめる。漏れや重なりがあれば止める。"""
    owners = {}
    for server, envs in SERVER_TASKS.items():
        for env in envs:
            owners.setdefault(env, []).append(server)
    problems = [f'{env} がどのサーバにも無い' for env in TASKS if env not in owners]
    problems += [f'{env} が {servers} の複数のサーバにある' for env, servers in owners.items() if len(servers) > 1]
    problems += [f'{env} は TASKS に無い' for env in owners if env not in TASKS]
    if problems:
        sys.exit('SERVER_TASKS の振り分けに問題がある: ' + '、'.join(problems))


def server_of(env):
    return next(server for server, envs in SERVER_TASKS.items() if env in envs)


def select(jobs, args):
    """--server と --env で一覧を絞る。"""
    if getattr(args, 'server', None) is not None:
        if args.server not in SERVER_TASKS:
            sys.exit(f'--server は {sorted(SERVER_TASKS)} のどれか')
        jobs = [j for j in jobs if j['env'] in SERVER_TASKS[args.server]]
    if getattr(args, 'env', None):
        jobs = [j for j in jobs if j['env'] in args.env]
    return jobs


def write_jobs(jobs):
    os.makedirs(QUEUE_DIR, exist_ok=True)
    with open(JOBS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader()
        w.writerows(jobs)


def read_jobs():
    if not os.path.exists(JOBS_CSV):
        sys.exit(f'{JOBS_CSV} がありません。先に `run_queue.py plan` を実行してください')
    with open(JOBS_CSV, newline='') as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------
# 終わったかどうかの判定と記録
# ----------------------------------------------------------------------

def is_done(job):
    """eval_metrics*.csv が揃っていれば終わっている。学習の最後にだけ書かれるファイルである。"""
    names = ['eval_metrics0.csv', 'eval_metrics1.csv'] if job['method'] == 'proposed' else ['eval_metrics.csv']
    return all(os.path.isfile(os.path.join(job['save_dir'], n)) for n in names)


def append_run(row):
    new = not os.path.exists(RUNS_CSV)
    with open(RUNS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_runs():
    if not os.path.exists(RUNS_CSV):
        return []
    with open(RUNS_CSV, newline='') as f:
        return list(csv.DictReader(f))


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ----------------------------------------------------------------------
# サブコマンド
# ----------------------------------------------------------------------

def cmd_plan(args):
    check_servers()
    jobs = build_jobs()
    write_jobs(jobs)
    print(f'{JOBS_CSV} に {len(jobs)} 本を書いた')
    count = {}
    for j in jobs:
        key = (j['phase'], j['method'], j['env'])
        count[key] = count.get(key, 0) + 1
    print(f'  {"段階":<12}{"手法":<10}{"タスク":<16}本数')
    for (phase, method, env), n in count.items():
        print(f'  {phase:<12}{method:<10}{env:<16}{n}')
    pending = [e for e, a in SAC_V1_ALPHA.items() if a is None]
    if pending:
        print(f'  SAC-v1 の最良 α が未定のタスク: {", ".join(pending)}。'
              'α 選びの結果を見て SAC_V1_ALPHA に書き、もう一度 plan を実行すると 2022〜2025 が加わる')
    print('  サーバごとの本数:')
    for server, envs in SERVER_TASKS.items():
        n = sum(1 for j in jobs if j['env'] in envs)
        print(f'    {server}番サーバ  {n:>3} 本  {", ".join(envs)}')
    done = sum(is_done(j) for j in jobs)
    print(f'  うち、このサーバで終わっているもの: {done} 本')


def cmd_run(args):
    check_servers()
    jobs = select(read_jobs(), args)
    todo = [j for j in jobs if not is_done(j)]
    host = socket.gethostname()
    print(f'[{now()}] 一覧 {len(jobs)} 本のうち、終わっていない {len(todo)} 本を同時に {args.workers} 本ずつ回す', flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    running = []  # (job, Popen, 開始時刻, ログのパス, ログのファイル)
    failed = []
    try:
        while todo or running:
            while todo and len(running) < args.workers:
                job = todo.pop(0)
                cmd = [sys.executable] + job['cmd'].split(' ')
                if args.cuda is not None:
                    cmd += ['--cuda', str(args.cuda)]
                log_path = os.path.join(LOG_DIR, f'{job["job_id"]}.log')
                log = open(log_path, 'w')
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                running.append((job, proc, time.time(), log_path, log))
                print(f'[{now()}] 開始 {job["job_id"]}', flush=True)
            time.sleep(args.poll)
            for item in list(running):
                job, proc, t0, log_path, log = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                log.close()
                ok = proc.returncode == 0 and is_done(job)
                status = 'done' if ok else 'failed'
                append_run(dict(job_id=job['job_id'], method=job['method'], env=job['env'], seed=job['seed'],
                                server=server_of(job['env']), host=host,
                                start=datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S'),
                                end=now(), seconds=round(time.time() - t0), returncode=proc.returncode,
                                status=status, cuda='' if args.cuda is None else args.cuda,
                                save_dir=job['save_dir'], log=log_path))
                if not ok:
                    failed.append(job['job_id'])
                print(f'[{now()}] {"終了" if ok else "失敗"} {job["job_id"]}  {round(time.time() - t0)} 秒  '
                      f'残り {len(todo)} 本', flush=True)
    except KeyboardInterrupt:
        # 止めたときは動いている実行も止める。CSV が無いので、次の run でもう一度回る
        for job, proc, t0, log_path, log in running:
            proc.terminate()
            log.close()
            append_run(dict(job_id=job['job_id'], method=job['method'], env=job['env'], seed=job['seed'],
                            server=server_of(job['env']), host=host,
                            start=datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S'),
                            end=now(), seconds=round(time.time() - t0), returncode='', status='interrupted',
                            cuda='' if args.cuda is None else args.cuda, save_dir=job['save_dir'], log=log_path))
        print(f'[{now()}] 中断した。動いていた {len(running)} 本を止めた', flush=True)
        raise
    print(f'[{now()}] すべて終わった。失敗 {len(failed)} 本' + (f': {", ".join(failed)}' if failed else ''), flush=True)


def cmd_status(args):
    jobs = select(read_jobs(), args)
    runs = read_runs()
    last = {}
    for r in runs:
        last[r['job_id']] = r  # 同じ実行を何度か回した場合は最後の記録を見る
    secs = {}
    for r in runs:
        if r['status'] == 'done':
            secs.setdefault((r['method'], r['env']), []).append(float(r['seconds']))

    print(f'  {"手法":<10}{"タスク":<16}{"終了":>5}{"失敗":>5}{"残り":>5}  1本の平均時間')
    total_left_h, unknown = 0.0, []
    groups = {}
    for j in jobs:
        groups.setdefault((j['method'], j['env']), []).append(j)
    for (method, env), js in groups.items():
        done = sum(is_done(j) for j in js)
        fail = sum(1 for j in js if not is_done(j) and last.get(j['job_id'], {}).get('status') == 'failed')
        left = len(js) - done
        s = secs.get((method, env))
        avg = f'{sum(s) / len(s) / 3600:.1f} 時間' if s else '未計測'
        if left and s:
            total_left_h += left * sum(s) / len(s) / 3600
        elif left:
            unknown.append(f'{method} {env}')
        print(f'  {method:<10}{env:<16}{done:>5}{fail:>5}{left:>5}  {avg}')
    print(f'  残りの時間の目安: 測れた分だけで約 {total_left_h / args.workers / 24:.1f} 日（同時に {args.workers} 本）')
    if unknown:
        print(f'  1本の時間がまだ分からないもの: {", ".join(unknown)}')
    failed = [j['job_id'] for j in jobs if not is_done(j) and last.get(j['job_id'], {}).get('status') == 'failed']
    if failed:
        print(f'  失敗した実行（ログは {LOG_DIR}/）: ' + ', '.join(failed))


def main():
    os.chdir(HERE)  # 相対パスをこのファイルのあるディレクトリ基準にする。main.py などもここから相対パスで保存する
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('plan', help='実行待ちの一覧を作る')
    r = sub.add_parser('run', help='一覧の実行を続けて回す')
    r.add_argument('--workers', type=int, default=2, help='同時に回す本数（既定 2）')
    r.add_argument('--cuda', type=int, default=None, help='main.py / proposed_main.py に渡す GPU 番号。-1 で CPU')
    r.add_argument('--server', type=int, default=None, help='SERVER_TASKS でこの番号のサーバに割り当てたタスクだけ回す')
    r.add_argument('--env', action='append', help='このタスクだけ回す。複数回指定できる')
    r.add_argument('--poll', type=float, default=10, help='終わったかを確かめる間隔（秒）')
    s = sub.add_parser('status', help='進み具合を表示する')
    s.add_argument('--server', type=int, default=None, help='この番号のサーバの分だけ表示する')
    s.add_argument('--env', action='append', help='このタスクだけ表示する')
    s.add_argument('--workers', type=int, default=2, help='残りの時間の目安の計算に使う同時実行の本数')
    args = p.parse_args()
    {'plan': cmd_plan, 'run': cmd_run, 'status': cmd_status}[args.command](args)


if __name__ == '__main__':
    main()
