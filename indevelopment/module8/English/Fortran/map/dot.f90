module dot_m
contains
  subroutine dot(A, B, C, m, n)
    implicit none
    real, intent(in) :: A(m,n)
    real, intent(in) :: B(m,n)
    real, intent(inout) :: C(m)
    integer, intent(in) :: m, n
    
    integer :: i, j
    
    do j = 1,m
       C(j) = 0.0
       do i = 1,n
          C(j) = C(j) + A(i,j) * B(i,j)
       end do
    end do
  end subroutine dot
end module dot_m

