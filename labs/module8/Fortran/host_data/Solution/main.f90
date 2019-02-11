program main

   use dot_m
   use dot_cuda_m

   integer :: i, j, m, n
   integer :: AllocateStatus
   real, allocatable :: A(:,:)
   real, allocatable :: B(:,:)
   real, allocatable :: C(:)
   real, allocatable :: D(:)

   m = 10
   n = 10

   write(*,*) 'Running with m,n: ', m, n

   ! alloc arrays (A to D)
   allocate( A(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( B(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( C(m), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( D(m), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   ! Create random arrays A and B
   call random_seed()
   call random_number(A)
   call random_number(B)
   do j = 1,n 
      do i = 1,m
          A(i,j) = int(A(i,j)*10.0)
          B(i,j) = int(B(i,j)*10.0)
      end do
   end do

   ! call dot product on host
   call dot(A, B, C, m, n)
   
!$acc data copyin(A,B) copyout(D)

!$acc host_data use_device(A,B,D)
   call dot_cuda(A, B, D, m, n)
!$acc end host_data

!$acc end data 

   ! check answers
   write(*,*) 'maxval(abs(C-D)): ', maxval(abs(C-D))

end program


