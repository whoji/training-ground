import subprocess
import time

count = 0
while 1:
    count += 1
    cmd_str = "echo %s > ct.txt" % count
    p = subprocess.Popen(cmd_str, shell=True)
    p.wait()
    time.sleep(1)
    print("[python] heartbeat: putong putong putong ~ %d [%s]" % (
        count,
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    )