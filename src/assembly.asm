#the following line is added to cope with the error:
#relocation R_X86_64_32S against `.data' can not be used when making a shared object; recompile with -fPIC
#DEFAULT REL

.globl zero, one, one_fifth, half
#for export;

.section .data
zero:
.double 0.0

one:
.double 1.0

one_fifth:
.double 0.2

half:
.double 0.5


.section .text
.globl relu, hard_sigmoid, sum8args, gcd_qword, gcd_long, gcd_int, gcd_dword, stosl

jmp_ret:
	ret

.ifdef linux
;determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_qword:
	mov %rdi, %rcx
	mov %rsi, %rdx
gcd_qword_linux:
	mov %rcx, %rax
	or %rdx, %rdx
	jz jmp_ret
	mov %rdx, %rcx
	mov $0, %rdx
	div %rcx
#rax = quo, rdx = rem;
	jmp gcd_qword_linux

gcd_long:
	mov %rdi, %rcx
	mov %rsi, %rdx
gcd_long_linux:
	mov %rcx, %rax
	or %rdx, %rdx
	jz jmp_ret
	mov %rdx, %rcx
	cqo
#Convert Quadword to Double Quadword, ie, Sign-extends the contents of RAX to RDX:RAX.
	idiv %rcx
	jmp gcd_long_linux

;determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_int:
	mov %edi, %ecx
	mov %esi, %edx
gcd_int_linux:
	mov %ecx, %eax
	or %edx, %edx
	jz jmp_ret
	mov %edx, %ecx
	cdq
# Convert Doubleword to Quadword, ie, Sign-extends register EAX and saves the results in register pair EDX:EAX.
	idiv %ecx
#eax = quo, edx = rem;
	jmp gcd_int_linux

;determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_dword:
	mov %edi, %ecx
	mov %esi, %edx
gcd_dword_linux:
	mov %ecx, %eax
	or %edx, %edx
	jz jmp_ret
	mov %edx, %ecx
	xor %edx, %edx
	div %ecx
#eax = quo, edx = rem;
	jmp gcd_dword_linux

stosl:
	mov %esi, %eax
	mov %rdx, %rcx
	rep stosl
	ret

sum8args:
	mov %rdi, %rax
	add %rsi, %rax
	add %rdx, %rax
	add %rcx, %rax
	add %r8, %rax
	add %r9, %rax
	add 8(%rsp), %rax
	add 16(%rsp), %rax
	ret


.else

#determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_qword:
	mov %rcx, %rax
	or %rdx, %rdx
	jz jmp_ret
	mov %rdx, %rcx
	mov $0, %rdx
	div %rcx
#rax = quo, rdx = rem;
	jmp gcd_qword

gcd_long:
	mov %rcx, %rax
	or %rdx, %rdx
	jz jmp_ret
	mov %rdx, %rcx
	cqo
#Convert Quadword to Double Quadword, ie, Sign-extends the contents of RAX to RDX:RAX.
	idiv %rcx
#rax = quo, rdx = rem;
	jmp gcd_long


#determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_int:
	mov %ecx, %eax
	or %edx, %edx
	jz jmp_ret
	mov %edx, %ecx
	cdq
# Convert Doubleword to Quadword, ie, Sign-extends register EAX and saves the results in register pair EDX:EAX.
	idiv %ecx
#eax = quo, edx = rem;
	jmp gcd_int

#determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_dword:
	mov %ecx, %eax
	or %edx, %edx
	jz jmp_ret
	mov %edx, %ecx
	xor %edx, %edx
	div %ecx
#eax = quo, edx = rem;
	jmp gcd_dword



stosl:
.ifdef _DEBUG
	mov %rcx, %rdi
	mov %edx, %eax
	mov %r8, %rcx
	rep stosl
.else
	push %rdi
	mov %rcx, %rdi
	mov %edx, %eax
	mov %r8, %rcx
	rep stosl
	pop %rdi
.endif
	ret


sum8args:
	mov %rcx, %rax
	add %rdx, %rax
	add %r8, %rax
	add %r9, %rax
	add 40(%rsp), %rax
	add 48(%rsp), %rax
	add 56(%rsp), %rax
	add 64(%rsp), %rax
	ret
.endif


relu:
	maxsd (zero), %xmm0
	ret

hard_sigmoid:
	mulsd (one_fifth), %xmm0
	#0.2x

	addsd (half), %xmm0
	#0.2x + 0.5

	minsd (one), %xmm0
	#min(y, 1)

	maxsd (zero), %xmm0
	#max(y, 0)
	ret

#https://blog.csdn.net/celerychen2009/article/details/8934972
#https://stackoverflow.com/questions/40820814/relocation-r-x86-64-32s-against-bss-can-not-be-used-when-making-a-shared-obj
#https://stackoverflow.com/questions/6093547/what-do-r-x86-64-32s-and-r-x86-64-64-relocation-mean
#http://www.csee.umbc.edu/~chang/cs313.f04/nasmdoc/html/nasmdoc8.html#section-8.2
#https://eli.thegreenplace.net/2011/11/03/position-independent-code-pic-in-shared-libraries/
#https://www.nasm.us/xdoc/2.11.02/html/nasmdoc6.html#section-6.2.1
#reference book: Apress.Modern.X86.Assembly.Language.Programming.32-bit.64-bit
#https://blog.csdn.net/sivolin/article/details/41895701
#https://www.cnblogs.com/volva/p/11814998.html
#https://blog.csdn.net/roger_ranger/article/details/78854348
