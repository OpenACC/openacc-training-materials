program main

   !use cudafor
   use distance_map_m
   use distance_map_acc_m

   integer :: i, j, m, n
   integer :: AllocateStatus
   real, allocatable :: A(:)
   real, allocatable :: B(:)
   real, allocatable :: C(:,:)
   real, allocatable :: D(:,:)

   m = 4098
   n = 4098

   ! allocate arrays
   allocate( A(m), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( B(n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( C(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   allocate( D(m,n), stat = AllocateStatus)
   IF (AllocateStatus /= 0) STOP "*** Not enough memory ***"

   ! initialize A and B
   ! Create random arrays A and B
   call random_seed()
   call random_number(A)
   call random_number(B)
   do i = 1,m
      A(i) = int(A(i)*10.0)
   enddo
   do i = 1,n
      B(i) = int(B(i)*10.0)
   end do

   ! Call distance_map (on host)
   call distance_map(A,B,C,m,n)

   ! Call distance_map (OpenACC)
   call distance_map_acc(A,B,D,m,n)

   ! Check answers
   write(*,*) 'maxval(abs(C-D)): ', maxval(abs(C-D))

end program

