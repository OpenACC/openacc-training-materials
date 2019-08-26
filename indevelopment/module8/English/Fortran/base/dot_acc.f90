module dot_acc_m
contains
   subroutine dot_acc(A, B, C, m, n)
      implicit none
      real, intent(in) :: A(m,n)
      real, intent(in) :: B(m,n)
      real, intent(inout) :: C(n)
      integer, intent(in) :: m, n
    
      real :: temp
      integer :: i, j
    
!$acc parallel loop gang private(temp) copyin(A,B) copyout(C)
      do j = 1,n
         temp = 0.0
!$acc loop vector reduction(+:temp)
         do i = 1,m
            temp = temp + A(i,j) * B(i,j)
         end do
         C(j) = temp
      end do

      return
  end subroutine dot_acc
  
end module dot_acc_m

