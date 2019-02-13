module distance_map_m
contains
   subroutine distance_map(A, B, C, m, n)
      implicit none
      real, intent(in) :: A(:)
      real, intent(in) :: B(:)
      real, intent(inout) :: C(:,:)
      integer, intent(in) :: m, n
      
      integer :: i, j
 
      do j = 1,m
         do i = 1,n
            C(i,j) = dist(A(j), B(i))
         end do
      end do
   end subroutine distance_map

   function dist(a, b)
      real :: a, b

      dist = sqrt(a*a + b*b)
      return 
   end function
end module distance_map_m
