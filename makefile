.PHONY: all clean install rebuild

all: 
	make -j8 -C Linux 

clean:
	make -C Linux clean
	
test:
	make -C Linux test

rebuild: clean all
	
install:
	make -j8 -C Linux install
	@echo "finish installing eigen.so"