import os
import re
import sys
from util import utility

def readFolder(rootdir, sufix='.cpp'):
    cpp = {}

    for name in os.listdir(rootdir):
        path = os.path.join(rootdir, name)

        if path.endswith(sufix):
            name = name[0:-len(sufix)]
            cpp[name] = path
        elif os.path.isdir(path):
            print('recursively reading Folder not implemented')

    return cpp


def pwd():
    pwd = os.path.dirname(__file__)
    if not pwd:
        return '.'
    return pwd

    
def workspace_pwd():
    _pwd = pwd()
    
    if _pwd == '.':
        return '..'        
    if _pwd.endswith('..'):
        return _pwd + '/..'
    _pwd = os.path.dirname(_pwd)
    if not _pwd:
        return '.'
    return _pwd


def LD_LIBRARY_PATH():
    javaFile = "LD_LIBRARY_PATH.java"
    classFile = "LD_LIBRARY_PATH.class"
    if not os.path.exists(classFile):
        javacode = """
public class LD_LIBRARY_PATH {
public static void main(String[] args) {
    System.out.println(System.getProperty("java.library.path"));
}
}
        """

        utility.Text(javaFile).write(javacode)
        execute("javac " + javaFile);
        
    assert os.path.exists('LD_LIBRARY_PATH.class')
    LD_LIBRARY_PATH = os.popen('java LD_LIBRARY_PATH').read()
    print('LD_LIBRARY_PATH =', LD_LIBRARY_PATH)

    LD_LIBRARY_PATH = LD_LIBRARY_PATH.split(':')
    for path in LD_LIBRARY_PATH:
        if os.path.exists(path):
            LD_LIBRARY_PATH = path
            break

    if isinstance(LD_LIBRARY_PATH, list):
        LD_LIBRARY_PATH = LD_LIBRARY_PATH[0]
        os.makedirs(LD_LIBRARY_PATH)
    return LD_LIBRARY_PATH


def JAVA_HOME():
    if 'JAVA_HOME' in os.environ:
        JAVA_HOME = os.environ['JAVA_HOME']
    else:
        print('JAVA_HOME is not specified!')
        JAVA_HOME = '/usr/lib/jvm/java-7-openjdk-amd64'

    print('JAVA_HOME =', JAVA_HOME)
    return JAVA_HOME


def makefile(file, target, prerequisites, command, prepend_tab=True):
#     if not prerequisites:
#         print(".PHONY : " + target, file=file)            
    print("%s: %s" % (target, prerequisites), file=file)
    if prepend_tab:
        command = '\t' + command 
    print(command, file=file)    


def cppbuild(eigen, password=None, clean=None):
#     sep = os.path.sep
    workspace_loc = workspace_pwd()
    print('workspace_loc =', workspace_loc)

    ProjDirPath = '%s/%s' % (workspace_loc, eigen)
    print('ProjDirPath =', ProjDirPath)
    
    cwd = os.getcwd()
    if cwd != ProjDirPath:
        os.chdir(ProjDirPath)
# now we are at '${workspace_loc}/eigen'

    obj2asm = readFolder('src', '.asm')
    print(obj2asm)
    
    with open('makefile', 'w') as file:
        print('so = lib%s.so' % eigen, file=file)
        print('artifact = obj/$(so)', file=file)        
        print('JAVA_HOME = ' + JAVA_HOME(), file=file)
        
        print('src_cpps = $(wildcard src/*.cpp)', file=file)
        print('src_cpp_objs = $(src_cpps:src/%.cpp=obj/%.o)', file=file)
        
        print('source_cpps = $(wildcard source/*.cpp)', file=file)
        print('source_cpp_objs = $(source_cpps:source/%.cpp=obj/%.o)', file=file)

        print('source_c99s = $(wildcard source/*.c)', file=file)
        print('source_c99_objs = $(source_c99s:source/%.c=obj/%.o)', file=file)

        print('asms = $(wildcard src/*.asm)', file=file)
        print('asm_objs = $(patsubst src/%.asm, obj/%.asm.o, $(wildcard src/*.asm))', file=file)
 
        print('gcc11 = g++ -Wall -std=c++11 -fPIC', file=file)
        print('gcc99 = gcc -w -std=c99 -fPIC', file=file)
        
        print('LD_LIBRARY_PATH = ' + LD_LIBRARY_PATH(), file=file)
        
        print('src_cpp_deps = $(src_cpps:src/%.cpp=src_cpp_dep/%.dep)', file=file)
        print('source_cpp_deps = $(source_cpps:source/%.cpp=source_cpp_dep/%.dep)', file=file)
        print('source_c99_deps = $(source_c99s:source/%.c=source_c99_dep/%.dep)', file=file)
        
        makefile(file, '$(LD_LIBRARY_PATH)/$(so)', '$(artifact) $(src_cpp_objs) $(asm_objs)', sudo('cp -f $(artifact) $(LD_LIBRARY_PATH)', password))
        makefile(file, '$(artifact)', '$(src_cpp_objs) $(source_cpp_objs) $(source_c99_objs) $(asm_objs)', '$(gcc11) -z noexecstack -shared -o $@ $^')
        
        print('-include $(src_cpp_deps)', file=file)
        print('-include $(source_cpp_deps)', file=file)
        print('-include $(source_c99_deps)', file=file)
        
        for obj_name, asm in obj2asm.items():  
            makefile(file, 'obj/%s.asm.o' % obj_name, asm, 'nasm -D__symbol=1 -Dlinux -f elf64 -o $@ $<')
            
    # -z noexecstack usage:
    # OpenJDK 64-Bit Server VM warning: You have loaded library /usr/local/cuda-10.0/lib64/libeigen.so 
    # which might have disabled stack guard. The VM will try to fix the stack guard now.
    # It's highly recommended that you fix the library with 'execstack -c <libfile>', or link it with '-z noexecstack'.
    # the final object must be put at the first line! 
        
        make_src_cpp_dep = """\
#\t@echo creating dependency file: $@        
\t@g++ -E -std=c++11 -MM -Iinc -Iinclude $< > $@
\t@sed -i '1 s#^#obj/&#g' $@
\t@echo '\t$$(gcc11) -Iinc -Iinclude -I$$(JAVA_HOME)/include -I$$(JAVA_HOME)/include/linux -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
\t@echo >> $@
#\t@cat $@
        """
        makefile(file, 'src_cpp_dep/%.dep', 'src/%.cpp', make_src_cpp_dep, False)        
 
        make_source_cpp_dep = """\
#\t@echo creating dependency file: $@        
\t@g++ -E -std=c++11 -MM -Iinclude $< > $@
\t@sed -i '1 s#^#obj/&#g' $@
\t@echo '\t$$(gcc11) -Iinclude -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
\t@echo >> $@
#\t@cat $@
        """
        makefile(file, 'source_cpp_dep/%.dep', 'source/%.cpp', make_source_cpp_dep, False)        

        make_source_c99_dep = """\
#\t@echo creating dependency file: $@        
\t@gcc -std=c99 -MM -Iinclude $< > $@
\t@sed -i '1 s#^#obj/&#g' $@
\t@echo '\t$$(gcc99) -Iinclude -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
\t@echo >> $@
#\t@cat $@
        """
        makefile(file, 'source_c99_dep/%.dep', 'source/%.c', make_source_c99_dep, False)        

        clean_src_cpp_dep = """\
ifneq (src_cpp_dep, $(wildcard src_cpp_dep))
\t@echo creating src_cpp_dep folder: src_cpp_dep
\tmkdir src_cpp_dep
else
\t@echo src_cpp_dep folder already exists!
endif        
\t-rm $(src_cpp_deps)
        """

        makefile(file, 'clean_src_cpp_dep', '', clean_src_cpp_dep, False)
        
        clean_source_cpp_dep = """\
ifneq (source_cpp_dep, $(wildcard source_cpp_dep))
\t@echo creating source_cpp_dep folder: source_cpp_dep
\tmkdir source_cpp_dep
else
\t@echo source_cpp_dep folder already exists!
endif        
\t-rm $(source_cpp_deps)
        """

        makefile(file, 'clean_source_cpp_dep', '', clean_source_cpp_dep, False)

        clean_source_c99_dep = """\
ifneq (source_c99_dep, $(wildcard source_c99_dep))
\t@echo creating source_cpp_dep folder: source_c99_dep
\tmkdir source_c99_dep
else
\t@echo source_c99_dep folder already exists!
endif        
\t-rm $(source_c99_deps)
        """

        makefile(file, 'clean_source_c99_dep', '', clean_source_c99_dep, False)

        create_src_cpp_dep = """\
\t@echo successfully created src_cpp_dep files        
ifneq (obj, $(wildcard obj))
\t@echo creating obj folder: obj
\tmkdir obj
\t@echo successfully created obj folder
else
\t@echo obj folder already exists!
endif
        """
        makefile(file, 'create_src_cpp_dep', '$(src_cpp_deps)', create_src_cpp_dep, False)
    
        create_source_cpp_dep = """\
\t@echo successfully created source_cpp_dep files        
ifneq (obj, $(wildcard obj))
\t@echo creating obj folder: obj
\tmkdir obj
\t@echo successfully created obj folder
else
\t@echo obj folder already exists!
endif
        """
        makefile(file, 'create_source_cpp_dep', '$(source_cpp_deps)', create_source_cpp_dep, False)

        create_source_c99_dep = """\
\t@echo successfully created source_c99_dep files        
ifneq (obj, $(wildcard obj))
\t@echo creating obj folder: obj
\tmkdir obj
\t@echo successfully created obj folder
else
\t@echo obj folder already exists!
endif
        """
        makefile(file, 'create_source_c99_dep', '$(source_c99_deps)', create_source_c99_dep, False)
        
        makefile(file, 'clean', '', '-rm $(src_cpp_objs) $(source_cpp_objs) $(source_c99_objs) $(asm_objs) $(artifact)')
        
    execute('cat makefile')
        
    if clean:
        execute('make clean')
    execute('make clean_src_cpp_dep && make create_src_cpp_dep')
    execute('make clean_source_cpp_dep && make create_source_cpp_dep')
    execute('make clean_source_c99_dep && make create_source_c99_dep')
    execute('make')
    
    if cwd != ProjDirPath:
        print('switch back to', cwd)
        os.chdir(cwd)

def sudo(cmd, password=None):
    if password is not None:
        return 'echo %s|sudo -S %s' % (password, cmd)
    return cmd

    
def execute(cmd, password=None):
    cmd = sudo(cmd, password)
    print(cmd)
    os.system(cmd)


# usage:
# python3 gunicorn.py --tcp=2000 --clean_gpu=0 --workers=4 --debug --worker-class=gevent
# python3 gunicorn.py --tcp=2000 --workers=4 --debug --worker-class=gevent
# python3 gunicorn.py --tcp=8000 --app=nlp_app --debug
# python3 gunicorn.py --cpp=eigen --password=qweasdzxc --tomcat --clean
if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) < 1:
        raise Exception('insufficient args to run!')

    clean_gpu = -1
    clean = None
    workers = 1
    debug = False
    stop = False
    worker_class = 'sync'
    app = None
    timeout = 86400
    cpp = None
    tcp = None
    password = None
    tomcat = None

    for s in sys.argv:
        if s.startswith('--'):
            s = s[2:]
        else:
            continue

        m = re.compile('tcp=(\d+)').fullmatch(s)
        if m:
            tcp = int(m.group(1))
            continue

        m = re.compile('clean_gpu=(\d+)').fullmatch(s)
        if m:
            clean_gpu = int(m.group(1))
            continue

        m = re.compile('workers=(\d+)').fullmatch(s)
        if m:
            workers = int(m.group(1))
            continue

        m = re.compile('debug').fullmatch(s)
        if m:
            debug = True
            continue

        m = re.compile('stop').fullmatch(s)
        if m:
            stop = True
            continue

        m = re.compile('timeout=(\d+)').fullmatch(s)
        if m:
            timeout = int(m.group(1))
            continue

        m = re.compile('worker-class=(\w+)').fullmatch(s)
        if m:
            worker_class = m.group(1)
            continue

        m = re.compile('app=(\w+)').fullmatch(s)
        if m:
            app = m.group(1)
            continue

        m = re.compile('cpp=(\w+)').fullmatch(s)
        if m:
            cpp = m.group(1)
            continue

        m = re.compile('password=(\w+)').fullmatch(s)
        if m:
            password = m.group(1)
            continue

        m = re.compile('tomcat').fullmatch(s)
        if m:
            tomcat = True
            continue
        
        m = re.compile('clean').fullmatch(s)
        if m:
            clean = True
            continue

    if cpp:
        cppbuild(cpp, password=password, clean=clean)

    if clean_gpu >= 0:
        res = os.popen('nvidia-smi').readlines()
        setPID = set()
        for s in res:
# Displayed as "C" for Compute Process, "G" for Graphics Process, and "C+G" for the process having both Compute and Graphics contexts.
            m = re.compile('\| +(\d+) +(\d+) +(\S+) +(\S+) +(\d+MiB) +\|').match(s)  # can not use fullmatch because these is an '\n' at the end of each line!
            if m:
                GPU = m.group(1)
                PID = m.group(2)
                Type = m.group(3)
                ProcessName = m.group(4)
                Usage = m.group(5)

                if Type == 'C' and int(GPU) == clean_gpu:
                    print('GPU = %s, PID = %s, Type = %s, ProcessName = %s, Usage = %s' % (GPU , PID , Type, ProcessName, Usage))
                    setPID.add(PID)
#             else:
#                 print(s)
#                 print('does not match!')
        for pid in setPID:
            execute('kill -9 ' + pid, password)

    if tcp and app:
        for pid in utility.get_process_pid(tcp):
            execute('kill -9 ' + pid)

        cwd = pwd()

        _cwd = os.path.relpath(os.getcwd())
        if _cwd != cwd:
            print('os.getcwd() =', _cwd)
            print('pwd() =', cwd)
            print("os.getcwd() != cwd, resetting cwd to " + cwd)
            os.chdir(cwd)

        accesslog = '../log/access%d.txt' % tcp
        errorlog = '../log/error%d.txt' % tcp

        if not os.path.exists(accesslog):
            print('%s does not exist, create a new txt file' % accesslog)
            utility.createNewFile(accesslog)
        elif os.path.getsize(accesslog) // 1024 // 1024 > 30:            
            with open(accesslog, 'w') as _:
                print('shrinking:', accesslog)

        if not os.path.exists(errorlog):
            print('%s does not exist, create a new txt file' % errorlog)
            utility.createNewFile(errorlog)
        elif os.path.getsize(errorlog) // 1024 // 1024 > 30:
            with open(errorlog, 'w') as _:
                print('shrinking:', errorlog)

        print('accesslog =', accesslog)
        print('errorlog =', errorlog)

        cmd = 'gunicorn --workers=%d '\
        '--bind=0.0.0.0:%d '\
        '--daemon '\
        '--timeout=%d '\
        '--access-logfile=%s '\
        '--error-logfile=%s '\
        '--log-level=debug '\
        '--capture-output '\
        '--worker-class=%s '\
        '%s:app' % (workers, tcp, timeout, accesslog, errorlog, worker_class, app)

        if stop:
            print('gunicorn is stopped!')
            execute('ps aux|grep python|grep -v grep|grep gunicorn.py|grep tcp=%s|cut -c 9-15|xargs kill -15' % tcp)
        else:
            execute(cmd)

            if debug:
                execute('tail -100f %s' % errorlog)

    if tomcat:
        execute('sh /usr/local/tomcat/bin/shutdown.sh')

        execute('sh /usr/local/tomcat/bin/startup.sh')

        execute('tail -100f /usr/local/tomcat/logs/catalina.out')
"""
import os

workers = 1

bind = '0.0.0.0:2000'

daemon = True

timeout = 864000

cwd = os.getcwd()
accesslog = os.path.join(cwd, 'log/access.txt')
errorlog = os.path.join(cwd, 'log/error.txt')

debug = True

#pidfile = os.path.join(cwd, 'app.pid')

logfile = os.path.join(cwd, 'log/log.txt')

loglevel = 'debug'

capture_output = True

"""

"""
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --log-syslog          Send *Gunicorn* logs to syslog. [False]
  --max-requests-jitter INT
                        The maximum jitter to add to the *max_requests*
                        setting. [0]
  --limit-request-fields INT
                        Limit the number of HTTP headers fields in a request.
                        [100]
  --reuse-port          Set the ``SO_REUSEPORT`` flag on the listening socket.
                        [False]
  -D, --daemon          Daemonize the Gunicorn process. [False]
  --proxy-protocol      Enable detect PROXY protocol (PROXY mode). [False]
  -w INT, --workers INT
                        The number of worker processes for handling requests.
                        [1]
  -u USER, --user USER  Switch worker processes to run as this user. [1004]
  --check-config        Check the configuration. [False]
  --reload-extra-file FILES
                        Extends :ref:`reload` option to also watch and reload
                        on additional files [[]]
  --backlog INT         The maximum number of pending connections. [2048]
  --reload              Restart workers when code changes. [False]
  --disable-redirect-access-to-syslog
                        Disable redirect access logs to syslog. [False]
  --error-logfile FILE, --log-file FILE
                        The Error log file to write to. [-]
  --pythonpath STRING   A comma-separated list of directories to add to the
                        Python path. [None]
  --proxy-allow-from PROXY_ALLOW_IPS
                        Front-end's IPs from which allowed accept proxy
                        requests (comma separate). [127.0.0.1]
  --access-logfile FILE
                        The Access log file to write to. [None]
  --log-config FILE     The log config file to use. [None]
  --log-config-dict LOGCONFIG_DICT
                        The log config dictionary to use, using the standard
                        Python [{}]
  --log-syslog-facility SYSLOG_FACILITY
                        Syslog facility name [user]
  --statsd-host STATSD_ADDR
                        ``host:port`` of the statsd server to log to. [None]
  --max-requests INT    The maximum number of requests a worker will process
                        before restarting. [0]
  --keep-alive INT      The number of seconds to wait for requests on a Keep-
                        Alive connection. [2]
  --certfile FILE       SSL certificate file [None]
  --preload             Load application code before the worker processes are
                        forked. [False]
  --paste STRING, --paster STRING
                        Load a PasteDeploy config file. The argument may
                        contain a ``#`` [None]
  --paste-global CONF   Set a PasteDeploy global config variable in
                        ``key=value`` form. [[]]
  --reload-engine STRING
                        The implementation that should be used to power
                        :ref:`reload`. [auto]
  -n STRING, --name STRING
                        A base to use with setproctitle for process naming.
                        [None]
  --suppress-ragged-eofs
                        Suppress ragged EOFs (see stdlib ssl module's) [True]
  --do-handshake-on-connect
                        Whether to perform SSL handshake on socket connect
                        (see stdlib ssl module's) [False]
  -g GROUP, --group GROUP
                        Switch worker process to run as this group. [1004]
  -m INT, --umask INT   A bit mask for the file mode on files written by
                        Gunicorn. [0]
  --cert-reqs CERT_REQS
                        Whether client certificate is required (see stdlib ssl
                        module's) [0]
  --limit-request-field_size INT
                        Limit the allowed size of an HTTP request header
                        field. [8190]
  --keyfile FILE        SSL key file [None]
  -c CONFIG, --config CONFIG
                        The Gunicorn config file. [None]
  --ciphers CIPHERS     Ciphers to use (see stdlib ssl module's) [TLSv1]
  -e ENV, --env ENV     Set environment variable (key=value). [[]]
  --log-syslog-prefix SYSLOG_PREFIX
                        Makes Gunicorn use the parameter as program-name in
                        the syslog entries. [None]
  --statsd-prefix STATSD_PREFIX
                        Prefix to use when emitting statsd metrics (a trailing
                        ``.`` is added, []
  -t INT, --timeout INT
                        Workers silent for more than this many seconds are
                        killed and restarted. [30]
  -R, --enable-stdio-inheritance
                        Enable stdio inheritance. [False]
  --worker-connections INT
                        The maximum number of simultaneous clients. [1000]
  -p FILE, --pid FILE   A filename to use for the PID file. [None]
  --log-level LEVEL     The granularity of Error log outputs. [info]
  --log-syslog-to SYSLOG_ADDR
                        Address to send syslog messages. [udp://localhost:514]
  -k STRING, --worker-class STRING
                        The type of workers to use. [sync]
  --graceful-timeout INT
                        Timeout for graceful workers restart. [30]
  --capture-output      Redirect stdout/stderr to specified file in
                        :ref:`errorlog`. [False]
  --no-sendfile         Disables the use of ``sendfile()``. [None]
  --chdir CHDIR         Chdir to specified directory before apps loading.
                        [/home/zlz/solution]
  --ssl-version SSL_VERSION
                        SSL version to use (see stdlib ssl module's)
                        [_SSLMethod.PROTOCOL_TLS]
  --ca-certs FILE       CA certificates file [None]
  --forwarded-allow-ips STRING
                        Front-end's IPs from which allowed to handle set
                        secure headers. [127.0.0.1]
  --spew                Install a trace function that spews every line
                        executed by the server. [False]
  --access-logformat STRING
                        The access log format. [%(h)s %(l)s %(u)s %(t)s
                        "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"]
  --worker-tmp-dir DIR  A directory to use for the worker heartbeat temporary
                        file. [None]
  -b ADDRESS, --bind ADDRESS
                        The socket to bind. [['127.0.0.1:8000']]
  --limit-request-line INT
                        The maximum size of HTTP request line in bytes. [4094]
  --threads INT         The number of worker threads for handling requests.
                        [1]
  --logger-class STRING
                        The logger you want to use to log events in Gunicorn.
                        [gunicorn.glogging.Logger]
  --initgroups          If true, set the worker process's group access list
                        with all of the [False]
                        
"""

# https://www.cnblogs.com/Liu-Jing/p/8298496.html
# https://blog.csdn.net/wjcapple/article/details/50071079
# https://blog.csdn.net/QQ1452008/article/details/50855810
# https://blog.csdn.net/baidu_38172402/article/details/88864517

