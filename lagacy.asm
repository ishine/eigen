section .text
global asm6args, asm8args, gcd_int
asm6args:
	mov rax, rdi; = 1
	add rax, rsi; =2
	add rax, rdx; = 3
	add rax, rcx; = 4
	add rax, r8; = 5
	add rax, r9; = 6
	ret

asm8args:
	mov rax, rdi; = 1
	add rax, rsi; =2
	add rax, rdx; = 3
	add rax, rcx; = 4
	add rax, r8; = 5
	add rax, r9; = 6
	add rax, [rsp]; = 7
	add rax, [rsp+8]; = 8
	ret

;determine the gcd of (rcx, rdx): gcd(rcx, rdx) = gcd(rdx, rcx % rdx)
gcd_int:
	mov eax, ecx
	or edx, edx
	jz jmp_ret
	mov ecx, edx
	cdq; Convert Doubleword to Quadword, ie, Sign-extends register EAX and saves the results in register pair EDX:EAX.
	idiv ecx; eax = quo, edx = rem;
	jmp gcd_int
jmp_ret:
	ret

;https://blog.csdn.net/sivolin/article/details/41895701
;https://www.cnblogs.com/volva/p/11814998.html
