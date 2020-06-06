target = eigen

artifact = lib$(target).so	

GCC = g++ -DNDEBUG -Wall -std=c++11 -fPIC -fopenmp
#GCC = g++ -DNDEBUG -Wall -std=c++11 -fopenmp

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

obj := src/main.o src/deeplearning/POSTagger.o src/deeplearning/NERTagger.o src/deeplearning/CWSTagger.o src/deeplearning/SyntaxParser.o src/deeplearning/Strassen.o src/deeplearning/classification.o src/deeplearning/keras.o src/deeplearning/bert.o src/deeplearning/matrix.o src/deeplearning/utility.o


$(artifact): $(obj)
	$(GCC) -z noexecstack -shared -o $@ $^ -L../../std/Linux -lstd -L../../hdf5/Linux -lhdf5 -L../../json/Linux -ljson -L../../sentencepiece/Linux -lsentencepiece

#target must be the first command to execute, so the include directive should follow it!
-include $(wildcard src/*.d) $(wildcard src/*/*.d) $(wildcard src/*/*/*.d)

$(target): src/main.o $(obj) 
	$(GCC) -o $@ $^ -ldl -L../../std/Linux -lstd -L../../hdf5/Linux -lhdf5 -L../../json/Linux -ljson -L../../sentencepiece/Linux -lsentencepiece  

src/%.o: ../src/%.cpp
	$(GCC) $(CMD)
	
src/%.o: ../src/%.cc
	$(GCC) $(CMD)

src/%.o: ../src/%.c
	gcc -DNDEBUG -w -std=c99 -fPIC $(CMD)
	        
clean:
	-rm -f $(target) src/*.d src/*.o src/*/*.d src/*/*.o *.class *.so

%.class: %.java
	@echo "compiling $< to $@"
	javac $<

install: $(artifact) LD_LIBRARY_PATH.class
	@echo "install $(artifact) to destination LD_LIBRARY_PATH"
	cp -f $(artifact) $(shell java -classpath ./ LD_LIBRARY_PATH)
	
	-mkdir -p ../../jetty/src/main/resources/lib/static/Linux/amd64
	@echo "install $(artifact) to destination ../../jetty/src/main/resources/lib/static/Linux/amd64"	
	cp $(artifact) ../../jetty/src/main/resources/lib/static/Linux/amd64
	@echo "install $(artifact) to destination ../../research-wordcloud/wordcloud-service/src/main/resources/lib/static/Linux/amd64"
	cp $(artifact) ../../research-wordcloud/wordcloud-service/src/main/resources/lib/static/Linux/amd64
	cd ../../research-wordcloud && git pull	&& git add . && git commit -m '$(artifact) modified' && git push

test: $(target)	
	./$(target) ../../jetty/src/main/resources
	@echo "finish testing $(target)!"

# https://blog.csdn.net/qq_42334372/article/details/83037362