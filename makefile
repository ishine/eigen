.PHONY: all clean install rebuild

all: 
	make -C Linux 

clean:
	make -C Linux clean
	
test:
	make -C Linux test

rebuild: clean all
	
install:
	make -C Linux install
	@echo "finish installing eigen.so"