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

