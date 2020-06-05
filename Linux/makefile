target = eigen

artifact = lib$(target).so	

GCC = g++ -DNDEBUG -Wall -std=c++11 -fPIC -fopenmp
#CMD = -c -fmessage-length=0 -O3 -MMD -MP -MF $(@:%.o=%.d) -MT $@ -o $@ $<
CMD = -c -fmessage-length=0 -O3 -MMD -MP -MT $@ -o $@ $<

c_src = $(wildcard ../src/*/*.c)
#$(info $(c_src))

cpp_src = $(wildcard ../src/*/*.cpp)
#$(info $(cpp_src))

cc_src = $(wildcard ../src/*/*.cc ../src/*/*/*.cc)
#$(info $(cc_src))

asm_src = $(wildcard ../src/*/*.asm)
#$(info $(asm_src))

obj = $(c_src:%.c=%.o) $(cpp_src:%.cpp=%.o) $(cc_src:%.cc=%.o) $(asm_src:%.asm=%.o)
obj := $(foreach file,$(obj),$(subst ../,,$(file)))
#$(info $(obj))

$(foreach file,$(sort $(dir $(obj))),$(if $(wildcard $(file)),,$(shell mkdir -p $(file))))

$(artifact): $(obj)
	$(GCC) -z noexecstack -shared -o $@ $^

#target must be the first command to execute, so the include directive should follow it!
-include $(wildcard src/*/*.d) $(wildcard src/*/*/*.d)

$(target): $(obj)
	$(GCC) -o $@ $^ -ldl

src/jvm/%.o: ../src/jvm/%.cpp
	$(GCC) -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux $(CMD)
	
src/%.o: ../src/%.cpp
	$(GCC) $(CMD)
	
src/%.o: ../src/%.cc
	$(GCC) $(CMD)

src/%.o: ../src/%.c
	gcc -DNDEBUG -w -std=c99 -fPIC $(CMD)
	        
src/%.o: ../src/%.asm
	as --defsym linux=1 -o $@ $<

clean:
	-rm -f src/*/*.d src/*/*.o *.class *.so

%.class: %.java
	@echo "compiling $< to $@"
	javac $<

install: $(artifact) LD_LIBRARY_PATH.class
#	@echo "install $(artifact) to destination LD_LIBRARY_PATH"
#	cp -f $(artifact) $(shell java -classpath ./ LD_LIBRARY_PATH)
	
	-mkdir -p ../../jetty/src/main/resources/lib/static/Linux/amd64
	@echo "install $(artifact) to destination ../../jetty/src/main/resources/lib/static/Linux/amd64"	
	cp $(artifact) ../../jetty/src/main/resources/lib/static/Linux/amd64
	@echo "install $(artifact) to destination ../../research-wordcloud/wordcloud-service/src/main/resources/lib/static/Linux/amd64"
	cp $(artifact) ../../research-wordcloud/wordcloud-service/src/main/resources/lib/static/Linux/amd64
	cd ../../research-wordcloud && git pull	&& git add . && git commit -m '$(artifact) modified' && git push

test: $(target)	
	./$(target) ../../
	@echo "finish testing $(target)!"

# https://blog.csdn.net/qq_42334372/article/details/83037362