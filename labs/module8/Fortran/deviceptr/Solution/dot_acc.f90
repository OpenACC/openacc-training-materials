module dot_acc_m
contains
   subroutine dot_acc(A, B, C, m, n)
      implicit none
      real, device, intent(in) :: A(m,n)
      real, device, intent(in) :: B(m,n)
      real, device, intent(inout) :: C(m)
      integer, intent(in) :: m, n

      integer :: i, j
      real temp
 
!$acc parallel loop gang private(temp) deviceptr(A, B, C)
      do j = 1,m
         temp = 0.0
!$acc loop vector reduction(+:temp)
         do i = 1,n
            temp = temp + A(i,j) * B(i,j)
         end do
         C(j) = temp
      end do

      return
   end subroutine dot_acc
 
end module dot_acc_m

