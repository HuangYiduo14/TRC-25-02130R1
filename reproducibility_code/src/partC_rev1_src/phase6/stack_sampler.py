"""Periodic actual Python stacks without the unsafe native timeout watchdog."""
import os
import sys
import threading
import time
import traceback


class StackSampler:
    def __init__(self,stream,interval=300):
        if interval<=0:raise ValueError("Positive stack cadence required")
        self.stream=stream;self.interval=interval;self.stop=threading.Event();self.thread=None;self.failure=None;self.samples=0

    def sample(self):
        # sys._current_frames returns strong frame references while the interpreter
        # lock is held; formatting stays in ordinary Python execution.
        frames=sys._current_frames();names={t.ident:t.name for t in threading.enumerate()}
        self.stream.write(f"\nPYTHON_STACK_SAMPLE unix={time.time():.6f} pid={os.getpid()} interval={self.interval}\n")
        for ident,frame in sorted(frames.items()):
            self.stream.write(f"THREAD {ident} {names.get(ident,'unknown')}\n")
            self.stream.writelines(traceback.format_stack(frame))
        self.stream.flush()
        self.samples+=1

    def __enter__(self):
        def run():
            while not self.stop.wait(self.interval):
                try:self.sample()
                except Exception as exc:self.failure=repr(exc);break
        self.thread=threading.Thread(target=run,name="python_stack_sampler",daemon=True);self.thread.start();return self

    def __exit__(self,*exc):
        self.stop.set();self.thread.join(10)
        if self.thread.is_alive():raise RuntimeError("Stack sampler failed to stop")
        if self.failure:raise RuntimeError("Stack sampler failure: "+self.failure)
