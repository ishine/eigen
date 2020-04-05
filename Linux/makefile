target = eigen

artifact = lib$(target).so	

GCC = g++ -Wall -std=c++11 -fPIC -fopenmp
#CMD = -c -fmessage-length=0 -O3 -MMD -MP -MF $(@:%.o=%.d) -MT $@ -o $@ $<
CMD = -c -fmessage-length=0 -O3 -MMD -MP -MT $@ -o $@ $<

FOLDERS = $(foreach dir,$(wildcard ../src/*),$(subst ../,,$(dir)))
$(info $(FOLDERS))

-include $(wildcard src/*/*.d)

$(foreach dir,$(FOLDERS),$(shell mkdir -p $(dir)))

C_OBJS = $(foreach dir, $(wildcard ../src/*/*.c),$(subst ../,,$(dir)))
C_OBJS := $(C_OBJS:%.c=%.o)
$(info $(C_OBJS))

CPP_OBJS = $(foreach dir, $(wildcard ../src/*/*.cpp),$(subst ../,,$(dir)))
CPP_OBJS := $(CPP_OBJS:%.cpp=%.o)
$(info $(CPP_OBJS))

ASM_OBJS = $(foreach dir, $(wildcard ../src/*/*.asm),$(subst ../,,$(dir)))
ASM_OBJS := $(ASM_OBJS:%.asm=%.o)
$(info $(ASM_OBJS))

obj = $(CPP_OBJS) $(C_OBJS) $(ASM_OBJS)

$(artifact): $(obj)
	$(GCC) -z noexecstack -shared -o $@ $^

$(target): $(obj)
	$(GCC) -o $@ $^ -ldl

src/jvm/%.o: ../src/jvm/%.cpp
	$(GCC) -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux $(CMD)
	
src/%.o: ../src/%.cpp
	$(GCC) $(CMD)
	
src/%.o: ../src/%.c
	gcc -w -std=c99 -fPIC $(CMD)
	        
src/%.o: ../src/%.asm
	as --defsym linux=1 -o $@ $<

clean:
	-rm -f src/*/*.d src/*/*.o *.class *.so

%.class: %.java
	@echo "compiling $< to $@"
	javac $<

install: $(artifact) LD_LIBRARY_PATH.class
	@echo "install $(artifact) to destination LD_LIBRARY_PATH"
	cp -f $(artifact) $(shell java -classpath ./ LD_LIBRARY_PATH)

test: $(target)	
	./$(target) ../../
	@echo "finish testing $(target)!"

# https://blog.csdn.net/qq_42334372/article/details/83037362