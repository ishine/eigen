target = eigen

artifact = lib$(target).so	

#GCC = g++ -DNDEBUG -Wall -std=c++11 -fPIC -fopenmp
GCC = g++ -DNDEBUG -Wall -std=c++11 -fopenmp

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
	$(GCC) -z noexecstack -shared -o $@ $^ -L../../std/Linux -lstd -L../../hdf5/Linux -lhdf5 -L../../json/Linux -ljson -L../../sentencepiece/Linux -lsentencepiece

#target must be the first command to execute, so the include directive should follow it!
	        
$(target):
#	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/POSTagger.o -o src/deeplearning/POSTagger.o ../src/deeplearning/POSTagger.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/matrix.o -o src/deeplearning/matrix.o ../src/deeplearning/matrix.cpp
#	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/NERTagger.o -o src/deeplearning/NERTagger.o ../src/deeplearning/NERTagger.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/CWSTagger.o -o src/deeplearning/CWSTagger.o ../src/deeplearning/CWSTagger.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/bert.o -o src/deeplearning/bert.o ../src/deeplearning/bert.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/utility.o -o src/deeplearning/utility.o ../src/deeplearning/utility.cpp
#	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/SyntaxParser.o -o src/deeplearning/SyntaxParser.o ../src/deeplearning/SyntaxParser.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/Strassen.o -o src/deeplearning/Strassen.o ../src/deeplearning/Strassen.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/keras.o -o src/deeplearning/keras.o ../src/deeplearning/keras.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/deeplearning/classification.o -o src/deeplearning/classification.o ../src/deeplearning/classification.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -c -fmessage-length=0 -O3 -MMD -MP -MT src/main.o -o src/main.o ../src/main.cpp
	g++ -DNDEBUG -Wall -std=c++11 -fopenmp -o eigen src/main.o src/deeplearning/Strassen.o \
	src/deeplearning/utility.o \
	src/deeplearning/matrix.o \
	src/deeplearning/CWSTagger.o \
	src/deeplearning/keras.o \
	src/deeplearning/classification.o \
	-ldl -L../../std/Linux -lstd \
	-L../../hdf5/Linux -lhdf5
#	src/deeplearning/NERTagger.o 
#	 
#	src/deeplearning/bert.o	 
#	src/deeplearning/SyntaxParser.o	 
#	 
#		
#	src/deeplearning/POSTagger.o
#  -L../../json/Linux -ljson -L../../sentencepiece/Linux -lsentencepiece

test: $(target)	
	./$(target) ../../jetty/src/main/resources
	@echo "finish testing $(target)!"

clean:
	-rm -rf $(target) src/* src/*/*.d src/*/*.o *.class *.so


# https://blog.csdn.net/qq_42334372/article/details/83037362