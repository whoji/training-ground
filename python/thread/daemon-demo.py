## based on this papge
## https://camcairns.github.io/python/2017/09/06/python_watchdog_jobs_queue.html

## for the thread stuff. refer to this
## https://blog.csdn.net/zhangzheng0413/article/details/41728869

import sys
import time
from queue import Queue
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class SqlLoaderWatchdog(PatternMatchingEventHandler):
    ''' Watches a nominated directory and when a trigger file is
        moved to take the ".trigger" extension it proceeds to execute
        the commands within that trigger file.

        Notes
        ============
        Intended to be run in the background
        and pickup trigger files generated by other ETL process
    '''

    def __init__(self, queue, patterns):
        PatternMatchingEventHandler.__init__(self, patterns=patterns)
        self.queue = queue

    def process(self, event):
        '''
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        '''
        self.queue.put(event)

    def on_moved(self, event):
        self.process(event)


def process_load_queue(q):
    '''This is the worker thread function. It is run as a daemon
       threads that only exit when the main thread ends.

       Args
       ==========
         q:  Queue() object
    '''
    while True:
        if not q.empty():
            event = q.get()
            now = datetime.datetime.utcnow()
            print ("{0} -- Pulling {1} off the queue ...".format(now.strftime("%Y/%m/%d %H:%M:%S"), event.dest_path))
            log_path = "./logging.txt"
            with open(log_path, "a") as f:
                f.write("{0} -- Processing {1}...\n".format(now.strftime("%Y/%m/%d %H:%M:%S"),event.dest_path))

            # read the contents of the trigger file
            cmd = "cat {0} | while read command; do $; done >> {1} 2>&1".format(event.dest_path, log_path)
            subprocess.call(cmd, shell=True)

            # once done, remove the trigger file
            os.remove(event.dest_path)

            # log the operation has been completed successfully
            now = datetime.datetime.utcnow()
            with open(log_path, "a") as f:
                f.write("{0} -- Finished processing {1}...\n".format(now.strftime("%Y/%m/%d %H:%M:%S"), event.dest_path))
        else:
            time.sleep(1)


if __name__ == '__main__':

    # create queue
    watchdog_queue = Queue()

    # Set up a worker thread to process database load
    worker = Thread(target=process_load_queue, args=(watchdog_queue,))
    worker.setDaemon(True)
    worker.start()

    # setup watchdog to monitor directory for trigger files
    args = sys.argv[1:]
    patt = ["*.trigger"]
    event_handler = SqlLoaderWatchdog(watchdog_queue, patterns=patt)
    observer = Observer()
    observer.schedule(event_handler, path=args[0] if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()