.section .text
.globl sum8args

.ifdef linux
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
