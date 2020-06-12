//the following line is added to cope with the error:
//relocation R_X86_64_32S against `.data' can not be used when making a shared object; recompile with -fPIC
//DEFAULT REL
//relocation R_X86_64_PC32 against symbol `zero' can not be used when making a shared object; recompile with -fPIC
//assembler settings:
//output file = std.o
//nasm -D__symbol=1 -f elf64 ${INPUTS} -o${PWD}/${OUTPUT}
.intel_syntax noprefix

.global zero, one, one_fifth, half
// for export
.section .data
zero:
.double 0.0
one:
.double 1.0
one_fifth:
.double 0.2
half:
.double 0.5

//reference book: Apress.Modern.X86.Assembly.Language.Programming.32-bit.64-bit
//https://blog.csdn.net/celerychen2009/article/details/8934972
//https://stackoverflow.com/questions/40820814/relocation-r-x86-64-32s-against-bss-can-not-be-used-when-making-a-shared-obj
//https://stackoverflow.com/questions/6093547/what-do-r-x86-64-32s-and-r-x86-64-64-relocation-mean
//http://www.csee.umbc.edu/~chang/cs313.f04/nasmdoc/html/nasmdoc8.html//section-8.2
//https://eli.thegreenplace.net/2011/11/03/position-independent-code-pic-in-shared-libraries/
//https://www.nasm.us/xdoc/2.11.02/html/nasmdoc6.html//section-6.2.1
//https://blog.csdn.net/sivolin/article/details/41895701
//https://www.cnblogs.com/volva/p/11814998.html
//https://blog.csdn.net/roger_ranger/article/details/78854348
//http://sourceware.org/binutils/docs-2.17/as/index.html
//https://www.cnblogs.com/binsys/articles/1303927.html
//https://www.ibm.com/developerworks/cn/linux/l-gas-nasm.html
