target = $(lastword $(subst /, , $(shell pwd)))
artifact = lib$(target).so

src_cpp = $(wildcard src/*/*.cpp)
src_c = $(wildcard src/*/*.c) 

GCC = g++ -Wall -std=c++11 -fPIC -fopenmp
CC = gcc -w -std=c99 -fPIC

obj = $(src_cpp:%.cpp=%.o) $(src_c:%.c=%.o) $(patsubst %.asm, %.o, $(wildcard src/*/*.asm))

$(artifact): $(obj)
	$(GCC) -z noexecstack -shared -o $@ $^

$(target): $(obj)
	$(GCC) -ldl -o $@ $^

src/jvm/%.cpp.dep: src/jvm/%.cpp
	@g++ -E -std=c++11 -MM $< > $@
#'&' denotes the previous matching pattern!	
	@sed -i '1 s#^#src/jvm/&#g' $@
	@echo '	$$(GCC) -I$$(JAVA_HOME)/include -I$$(JAVA_HOME)/include/linux -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@

src/%.cpp.dep: src/%.cpp
	@echo -n $< > $@
	@sed -r -i 's#[^/]+\.cpp##g' $@
	@g++ -E -std=c++11 -MM $< >> $@
	@echo '	$$(GCC) -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@

src/%.c.dep: src/%.c
	@echo -n $< > $@
	@sed -r -i 's#[^/]+\.c##g' $@        
	@gcc -std=c99 -MM $< >> $@
	@echo '	$$(CC) -c -fmessage-length=0 -O3 -o $$@ $$<' >> $@
	@echo >> $@
	@cat $@
        
-include $(src_cpp:%.cpp=%.cpp.dep) $(src_c:%.c=%.c.dep)

src/%.o: src/%.asm
	as --defsym linux=1 -o $@ $<

clean:
	-rm -f src/*/*.dep src/*/*.o *.class *.so

%.class: %.java
	@echo "compiling $< to $@"
	javac $<

install: $(artifact) LD_LIBRARY_PATH.class
	@echo "install $(artifact) to destination LD_LIBRARY_PATH"
	cp -f $(artifact) $(shell java -classpath ./ LD_LIBRARY_PATH)

test: $(target)
	@echo "testing $(target):"
	./$(target)

# https://blog.csdn.net/qq_42334372/article/details/83037362