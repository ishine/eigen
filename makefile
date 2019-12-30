artifact = lib$(lastword $(subst /, , $(shell pwd))).so

src_cpp = $(wildcard src/*.cpp)
src_cpp_obj = $(src_cpp:src/%.cpp=src/%.o)

source_cpp = $(wildcard source/*.cpp)
source_cpp_obj = $(source_cpp:source/%.cpp=source/%.o)

source_c = $(wildcard source/*.c)
source_c_obj = $(source_c:source/%.c=source/%.o)

asm_obj = $(patsubst src/%.asm, src/%.o, $(wildcard src/*.asm))

src_cpp_dep = $(src_cpp:src/%.cpp=src/%.cpp.dep)
source_cpp_dep = $(source_cpp:source/%.cpp=source/%.cpp.dep)
source_c_dep = $(source_c:source/%.c=source/%.c.dep)

gpp = g++ -Wall -std=c++11 -fPIC
gcc = gcc -w -std=c99 -fPIC

$(artifact): $(src_cpp_obj) $(source_cpp_obj) $(source_c_obj) $(asm_obj)
	$(gpp) -z noexecstack -shared -o $@ $^

src/%.cpp.dep: src/%.cpp
	@g++ -E -std=c++11 -MM -Iinc -Iinclude $< > $@
	@sed -i '1 s#^#src/&#g' $@
	@echo '	$$(gpp) -Iinc -Iinclude -I$$(JAVA_HOME)/include -I$$(JAVA_HOME)/include/linux -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@
        
source/%.cpp.dep: source/%.cpp        
	@g++ -E -std=c++11 -MM -Iinclude $< > $@
	@sed -i '1 s#^#source/&#g' $@
	@echo '	$$(gpp) -Iinclude -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@
        
source/%.c.dep: source/%.c        
	@gcc -std=c99 -MM -Iinclude $< > $@
	@sed -i '1 s#^#source/&#g' $@
	@echo '	$$(gcc) -Iinclude -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@
        
-include $(src_cpp_dep)
-include $(source_cpp_dep)
-include $(source_c_dep)

src/%.o: src/%.asm
	as --defsym linux=1 -o $@ $<

clean:
	-rm -f src/*.dep source/*.dep 
	-rm -f src*.o source/*.o
	-rm -f *.class
	-rm -f *.so

%.class: %.java
	@echo "compiling $< to $@"
	javac $<

install: $(artifact) LD_LIBRARY_PATH.class
	@echo "install $(artifact) to destination LD_LIBRARY_PATH"
	cp -f $(artifact) $(shell java -classpath ./ LD_LIBRARY_PATH)

# make && make install
