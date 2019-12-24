import os
import re
import sys


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
        with open(java, 'w', encoding='utf8') as file:
            print(javacode, file=file)
        execute("javac " + javaFile)
        
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


def cppbuild(eigen, password=None, clean=None, exe=None):
#     sep = os.path.sep
    workspace_loc = workspace_pwd()
    print('workspace_loc =', workspace_loc)

    ProjDirPath = '%s/%s' % (workspace_loc, eigen)
    print('ProjDirPath =', ProjDirPath)
    
    cwd = os.getcwd()
    if cwd != ProjDirPath:
        os.chdir(ProjDirPath)
# now we are at '${workspace_loc}/eigen'
    try:
        obj2asm = readFolder('src', '.asm')
        print(obj2asm)
    except Exception as e:
        print(e)
        obj2asm = None
    
    with open('makefile', 'w') as file:        
        if exe:
            print('target = %s.exe' % eigen, file=file)
        else:
            print('target = lib%s.so' % eigen, file=file)
            
        print('artifact = obj/$(target)', file=file)        
        print('JAVA_HOME = ' + JAVA_HOME(), file=file)
        
        print('src_cpps = $(wildcard src/*.cpp)', file=file)
        print('src_cpp_objs = $(src_cpps:src/%.cpp=obj/%.o)', file=file)
        
        print('source_cpps = $(wildcard source/*.cpp)', file=file)
        print('source_cpp_objs = $(source_cpps:source/%.cpp=obj/%.o)', file=file)

        print('source_c99s = $(wildcard source/*.c)', file=file)
        print('source_c99_objs = $(source_c99s:source/%.c=obj/%.o)', file=file)

        if obj2asm:
            print('asms = $(wildcard src/*.asm)', file=file)
            print('asm_objs = $(patsubst src/%.asm, obj/%.asm.o, $(wildcard src/*.asm))', file=file)
        else:
            print('asm_objs = ', file=file)
 
        print('src_cpp_deps = $(src_cpps:src/%.cpp=src_cpp_dep/%.dep)', file=file)
        print('source_cpp_deps = $(source_cpps:source/%.cpp=source_cpp_dep/%.dep)', file=file)
        print('source_c99_deps = $(source_c99s:source/%.c=source_c99_dep/%.dep)', file=file)
        
        if exe:
            print('gcc11 = g++ -Wall -std=c++11', file=file)
            print('gcc99 = gcc -w -std=c99', file=file)
            makefile(file, '$(artifact)', '$(src_cpp_objs) $(source_cpp_objs) $(source_c99_objs) $(asm_objs)', '$(gcc11) -ldl -o $@ $^')
# add -ldl option to resolve the following linking errors:            
# undefined reference to `dlopen'
# undefined reference to `dlerror'
# undefined reference to `dlsym'
# undefined reference to `dlsym'
# undefined reference to `dlsym'
# undefined reference to `dlclose'            
        else:
            print('gcc11 = g++ -Wall -std=c++11 -fPIC', file=file)
            print('gcc99 = gcc -w -std=c99 -fPIC', file=file)        
            print('LD_LIBRARY_PATH = ' + LD_LIBRARY_PATH(), file=file)

            makefile(file, '$(LD_LIBRARY_PATH)/$(target)', '$(artifact) $(src_cpp_objs) $(asm_objs)', sudo('cp -f $(artifact) $(LD_LIBRARY_PATH)', password))        
            makefile(file, '$(artifact)', '$(src_cpp_objs) $(source_cpp_objs) $(source_c99_objs) $(asm_objs)', '$(gcc11) -z noexecstack -shared -o $@ $^')        
# -z noexecstack usage:
# OpenJDK 64-Bit Server VM warning: You have loaded library /usr/local/lib64/libeigen.so 
# which might have disabled stack guard. The VM will try to fix the stack guard now.
# It's highly recommended that you fix the library with 'execstack -c <libfile>', or link it with '-z noexecstack'.
# the final object must be put at the first line! 
        
        print('-include $(src_cpp_deps)', file=file)
        print('-include $(source_cpp_deps)', file=file)
        print('-include $(source_c99_deps)', file=file)
        
        if obj2asm:
            for obj_name, asm in obj2asm.items():  
                makefile(file, 'obj/%s.asm.o' % obj_name, asm, 'nasm -D__symbol=1 -Dlinux -isrc -f elf64 -o $@ $<')
        
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
# python3 makefile.py --cpp=eigen --password=123456 --clean --exe
if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) < 1:
        raise Exception('insufficient args to run!')

    clean = None
    cpp = None
    password = None
    exe = None

    for s in sys.argv:
        if s.startswith('--'):
            s = s[2:]
        else:
            continue

        m = re.compile('cpp=(\w+)').fullmatch(s)
        if m:
            cpp = m.group(1)
            continue

        m = re.compile('password=(\w+)').fullmatch(s)
        if m:
            password = m.group(1)
            continue
        
        m = re.compile('clean').fullmatch(s)
        if m:
            clean = True
            continue

        m = re.compile('exe').fullmatch(s)
        if m:
            exe = True
            continue
    if cpp:
        cppbuild(cpp, password=password, clean=clean, exe=exe)
