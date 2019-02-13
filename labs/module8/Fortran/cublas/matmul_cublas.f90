module matmul_cublas_m
contains
   subroutine matmul_cublas(A, B, D, m, n, k)
      use cudafor
      use cublas

      integer :: m, n, k
!      integer :: AllocateStatus
      !real, device, allocatable :: A(:,:)
      !real, device, allocatable :: B(:,:)
      !real, device, allocatable :: D(:,:)

      real, device, intent(in) :: A(:,:)
      real, device, intent(in) :: B(:,:)
      real, device, intent(out) :: D(:,:)


!      real, device, allocatable :: A_d(:,:)
!      real, device, allocatable :: B_d(:,:)
!      real, device, allocatable :: D_d(:,:)

      
      ! sourced allocation here allocates A_d to be the same size as A, and initializes
      ! the contents of A_d with values in A, i.e. is does the host2device transfer
!      allocate( A_d, source = A, stat = AllocateStatus)
!      IF (AllocateStatus /= 0) STOP "*** Not enough memory for A_d on device ***"
!      allocate( B_d, source = B, stat = AllocateStatus)
!      IF (AllocateStatus /= 0) STOP "*** Not enough memory for B_d on device ***"
      ! allocation with mold= just does the allocation but not the initialization
!      allocate( D_d, mold = D, stat = AllocateStatus)
!      IF (AllocateStatus /= 0) STOP "*** Not enough memory for D_d on device ***"

! CALL SGEMM(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, ldc)
!
! l = rows in C
! n = number of cols in C
! m = cols in A (if 'n')
! alpha = scalar alpha
! a is the matrix A - A(l,m)
! lda = leading dimension of A (lda >= n)
! b is the matrix B - B(m,n)
! ldb leading dimension of b (ldb >= m)
! beta = scalar beta
! c =C(l,n)
! ldc = leading dimension of c

      call sgemm('n','n', m, n, k, 1.0, A, m, B, k, 0.0, D, m)

!      D = D_d

      return
   end subroutine
end module

