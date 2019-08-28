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

