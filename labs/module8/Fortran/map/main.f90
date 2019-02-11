program main

   use cudafor
   use dot_m
   use dot_acc_m

   integer :: i, j, m, n
   integer :: AllocateStatus
   real, allocatable :: A(:,:)
   real, allocatable :: B(:,:)
   real, allocatable :: C(:)
   real, allocatable :: D(:)
   real, allocatable :: E(:)

   real, device, allocatable :: A_d(:,:)
   real, device, allocatable :: B_d(:,:)
   real, device, allocatable :: C_d(:)

   m = 4098
   n = 4098

   write(*,*) 'Running with m,n: ', m, n

   ! Similiar to deviceptr
   ! ---------------------

   ! malloc A to D
   ! Allocate arrays using Fortran allocate
   !allocate( A(m,n), stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   !allocate( B(m,n), stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   !allocate( C(m), stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   !allocate( D(m), stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   ! Random initialize A, B

   ! malloc device

   ! Copy A, B to device arrays
   ! sourced allocation here allocates A_d to be the same size as A, and initializes
   ! the contents of A_d with values in A, i.e. is does the host2device transfer
   !allocate( A_d, source = A, stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory for A_d on device ***"
   !allocate( B_d, source = B, stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory for B_s on device ***"
   ! allocation with mold= just does the allocation but no the initialization
   !allocate( C_d, mold = C, stat = AllocateStatus)
   !IF (AllocateStatus /= 0) STOP "*** Not enough memory for C_d on device ***"

   allocate(A(m,n))
   allocate(B(m,n))
   allocate(C(m))
   allocate(D(m))
   allocate(A_d(m,n))
   allocate(B_d(m,n))
   allocate(C_d(m))

   A(:) = 1
   B(:) = 0
   A_d = A
   B_d = B

   call dot_acc(A_d,B_d,C_d,m,n)
   C = C_d

   call dot(A,B,D,m,n)

   ! check answers
   write(*,*) 'maxval(abs(C-D)): ', maxval(abs(C-D))

end program


