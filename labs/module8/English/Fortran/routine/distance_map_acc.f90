module distance_map_acc_m
   use  dist_cuda_m
   use cudafor

contains
   subroutine distance_map_acc(A, B, C, m, n)
      implicit none
      real, intent(in) :: A(:)
      real, intent(in) :: B(:)
      real, intent(inout) :: C(:,:)
      integer, intent(in) :: m, n
         
      integer :: i, j

!$acc parallel loop copyin(A, B) copyout(C)
      do j = 1,m
!$acc loop
         do i = 1,n
            C(i,j) = dist_cuda(A(j), B(i) )
         end do
      end do
   end subroutine distance_map_acc
end module distance_map_acc_m

