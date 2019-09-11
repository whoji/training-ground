# script taken from
# https://www.michaelcho.me/article/using-pythons-watchdog-to-monitor-changes-to-a-directory
# also official doc
# https://pypi.org/project/watchdog/

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Watcher:
    DIRECTORY_TO_WATCH = './'

    def  __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler,
            self.DIRECTORY_TO_WATCH,
            recursive= True)

        self.observer.start()
        print("Observer started ...")

        try:
            while 1:
                time.sleep(5)
        except:
            self.observer.stop()
            print("ERROR. observer stopped")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            #Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
        elif event.event_type == 'modified':
            #Take any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)

if __name__ == '__main__':
    w = Watcher()
    w.run()