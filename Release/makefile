target = eigen

artifact = $(target).dll

GCC = g++ -Wall -std=c++11 -fPIC -fopenmp
#CMD = -c -fmessage-length=0 -O3 -MMD -MP -MF $(@:%.o=%.d) -MT $@ -o $@ $<
CMD = -c -fmessage-length=0 -O3 -MMD -MP -MT $@ -o $@ $<

SHELL=cmd.exe
FOLDERS = $(foreach dir,$(wildcard ../src/*),$(subst ../,,$(dir)))
#$(info $(FOLDERS))

-include $(wildcard src/*/*.d)

$(foreach dir,$(FOLDERS),$(shell mkdir $(subst /,\,$(dir))))

C_OBJS = $(foreach dir, $(wildcard ../src/*/*.c),$(subst ../,,$(dir)))
C_OBJS := $(C_OBJS:%.c=%.o)
#$(info $(C_OBJS))

CPP_OBJS = $(foreach dir, $(wildcard ../src/*/*.cpp),$(subst ../,,$(dir)))
CPP_OBJS := $(CPP_OBJS:%.cpp=%.o)
#$(info $(CPP_OBJS))

ASM_OBJS = $(foreach dir, $(wildcard ../src/*/*.asm),$(subst ../,,$(dir)))
ASM_OBJS := $(ASM_OBJS:%.asm=%.o)
#$(info $(ASM_OBJS))

obj = $(CPP_OBJS) $(C_OBJS) $(ASM_OBJS)

$(artifact): $(obj)
	$(GCC) -shared -o $@ $^

$(target).exe: $(obj)
	$(GCC) -o $@ $^

src/jvm/%.o: ../src/jvm/%.cpp
	$(GCC) -I"$(JAVA_HOME)/include" -I"$(JAVA_HOME)/include/win32" $(CMD)	
	
src/%.o: ../src/%.cpp
	$(GCC) $(CMD)
	
src/%.o: ../src/%.c
	gcc -w -std=c99 -fPIC $(CMD)
	        
src/%.o: ../src/%.asm
	as -o $@ $<

clean:
	-rd /s /q src

install: $(artifact) 
	@echo "install $(artifact) to destination ${MINGW_HOME}\bin"
	xcopy $(artifact) "${MINGW_HOME}\bin" /Y

test: $(target).exe
	./$(target).exe ../../
	@echo "finish testing $(target)!"	
