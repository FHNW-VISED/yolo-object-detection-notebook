"""Headless runner: jupyter_client based (nbclient lost the idle message on widget cells)."""
import sys, json, time, queue, os
import nbformat
from nbformat.v4 import output_from_msg
from jupyter_client.manager import start_new_kernel
src, out, timing_out = sys.argv[1:4]
CELL_TIMEOUT = int(os.environ.get('CELL_TIMEOUT', 900))
nb = nbformat.read(src, as_version=4)
import re
settings = sys.argv[4] if len(sys.argv) > 4 else None
if settings:
    cell = next(c for c in nb.cells if c.cell_type == 'code' and re.search(r'^DATASET_PATH\s*=', c.source, re.M))
    for line in settings.split(';'):
        name = line.split('=')[0].strip()
        cell.source = re.sub(rf'^{name}\s*=.*$', line.strip(), cell.source, flags=re.M)
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '2'
os.environ.pop('MPLBACKEND', None); os.environ.pop('NB_SMOKE', None)
km, kc = start_new_kernel(kernel_name='python3', cwd=os.getcwd())
times = {}; t0 = time.time(); n = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    code = '\n'.join(('# [audit-disabled] ' + l) if l.strip().startswith('%pip') else l for l in c.source.split('\n'))
    mid = kc.execute(code)
    n += 1; c.execution_count = n; c.outputs = []
    t = time.time(); got_reply = None; idle = False; timed_out = False
    while True:
        if time.time() - t > CELL_TIMEOUT:
            timed_out = True; km.interrupt_kernel(); time.sleep(3); break
        try:
            m = kc.get_iopub_msg(timeout=1)
        except queue.Empty:
            if got_reply is None:
                try:
                    r = kc.get_shell_msg(timeout=0.01)
                    if r['parent_header'].get('msg_id') == mid: got_reply = time.time()
                except queue.Empty: pass
            elif time.time() - got_reply > 5: break  # reply seen, idle lost
            continue
        if m['parent_header'].get('msg_id') != mid: continue
        mt = m['msg_type']; ct = m['content']
        if mt == 'status' and ct['execution_state'] == 'idle': idle = True; break
        if mt in ('stream', 'display_data', 'execute_result', 'error'):
            c.outputs.append(output_from_msg(m))
        elif mt == 'update_display_data':
            did = ct.get('transient', {}).get('display_id')
            for o in c.outputs:
                if o.get('output_type') == 'display_data' and o.get('metadata', {}).get('_did') == did:
                    o['data'] = ct['data']
        elif mt == 'clear_output':
            pass  # inside widgets this clears only the Output widget, not the cell
    if timed_out:
        c.outputs.append(nbformat.v4.new_output('error', ename='AuditTimeout', evalue=f'cell exceeded {CELL_TIMEOUT}s and was interrupted', traceback=[]))
    # drain shell reply
    if got_reply is None:
        try: kc.get_shell_msg(timeout=2)
        except queue.Empty: pass
    times[i] = time.time() - t
    print(f'cell {i} {times[i]:.1f}s' + (' ERROR' if any(o.output_type == 'error' for o in c.outputs) else ''), flush=True)
    json.dump(times, open(timing_out, 'w'))
    nbformat.write(nb, out)
nbformat.write(nb, out)
print('TOTAL', time.time() - t0)
km.shutdown_kernel(now=True)
