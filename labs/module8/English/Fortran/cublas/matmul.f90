module matmul_m
contains
  subroutine matmul(A, B, C, m, n, k)
    implicit none
    real, intent(in) :: A(m,k)
    real, intent(in) :: B(k,n)
    real, intent(inout) :: C(m,n)
    integer, intent(in) :: m, n, k
   
    integer :: i, j, l
    
    do i = 1, m
       do j = 1, n
          do l = 1, k
             C(i,j) = C(i,j) + A(i,l) * B(l,j)
          enddo
       end do
    end do
  end subroutine matmul
end module matmul_m

