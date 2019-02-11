module dot_acc_m
contains
   subroutine dot_acc(A, B, C, m, n)
      implicit none
      real, intent(in) :: A(m,n)
      real, intent(in) :: B(m,n)
      real, intent(inout) :: C(m)
      integer :: m, n

      integer :: i, j
      real temp
 
!$acc parallel loop gang private(temp) copyin(A,B) copyout(C)
      do j = 1,m
         temp = 0.0
!$acc loop vector reduction(+:temp)
         do i = 1,n
            temp = temp + A(i,j) * B(i,j)
         end do
         C(j) = temp
      end do
   end subroutine dot_acc
 
end module dot_acc_m

