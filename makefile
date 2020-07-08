.PHONY: all clean install rebuild

cpu_count =$(shell cat /proc/cpuinfo | grep processor | wc -l)

$(info cpu_count = $(cpu_count))

ifneq ($(shell echo $(cpu_count) | awk '{if($$1 >= "1") {print $$1;}}'), $(nullstring))
	paralleled :=-j$(cpu_count)
endif
	paralleled :=

$(info paralleled = $(paralleled))

all: 
	make $(paralleled) -C Linux 

clean:
	make -C Linux clean
	
test:
	make -C Linux test

rebuild: clean all
	
install:
	make -C Linux install
	@echo "finish installing eigen.so"