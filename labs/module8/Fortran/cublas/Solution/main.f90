program main

   !use cudafor
   !use cublas
   use matmul_m
   use matmul_cublas_m

   integer :: i, j, m, n, k
   integer :: AllocateStatus
   real, allocatable :: A(:,:)   ! A(m,k)
   real, allocatable :: B(:,:)   ! B(k,n)
   real, allocatable :: C(:,:)   ! C(m,n)
   real, allocatable :: D(:,:)   ! D(m,n)

   m = 2
   n = 4
   k = 3
   
   write(*,*) 'Running with m,n,k: ', m, n, k


   allocate( A(m,k), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( B(k,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( C(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( D(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"


   do i=1,m
      do j=1,k
         A(i,j) = real(i+j)
      enddo
   enddo

   do i=1,k
      do j=1,n
         B(i,j) = real(i+j)
      enddo
   enddo

!$acc enter data copyin(A(:), B(:)) create(D(:))

   call matmul(A, B, C, m, n, k);

!$acc host_data use_device(A, B, D)
   call matmul_cublas(A, B, D, m, n, k)
!$acc end host_data

!$acc exit data delete(A, B) copyout(D(:))
      
   ! Check accuracy
   write(*,*) 'maxval(abs(C-D)): ',maxval(abs(C-D))

   write(*,*) "Program finished sucessfully."

end program
