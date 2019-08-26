!
! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!
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

