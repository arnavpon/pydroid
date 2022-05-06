# ***

import time,threading,ctypes,inspect

def _async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res == 1:
        print("Found valid thread")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("Timeout Exception")

def stop_thread(thread):
    print("[script.py] Stopping thread #{0}".format(thread.ident))
    _async_raise(thread.ident, SystemExit)
    
def text_thread_run(code):
    print("[script.py] Executing code...")
    try:
        env={}
        exec(code, env, env)
    except Exception as e:
        print(e)
    # issue here is we don't have async/await so we don't know when the code is complete
    # the workaround being used is to run a while loop w/ a timeout, which won't work for us
    
#   This is the code to run Text functions...
def mainTextCode(code):
    print("[script.py] main...")
    global thread1
    thread1 = threading.Thread(target=text_thread_run, args=(code,),daemon=True)
    thread1.start()
    timeout = 15 # change timeout settings in seconds here...
    # remove timeout b/c we don't know how long execution will take
    # will need to ensure only 1 thread is active a time to avoid leaks
    thread1_start_time = time.time()
    while thread1.is_alive():
        if time.time() - thread1_start_time > timeout:
            stop_thread(thread1)
            raise TimeoutError
        time.sleep(1)