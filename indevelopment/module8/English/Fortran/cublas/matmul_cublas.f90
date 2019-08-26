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
module matmul_cublas_m
contains
   subroutine matmul_cublas(A, B, D, m, n, k)
      use cudafor
      use cublas

      integer :: m, n, k

      real, device, intent(in) :: A(:,:)
      real, device, intent(in) :: B(:,:)
      real, device, intent(out) :: D(:,:)

! CALL SGEMM(transa, transb, l, n, m, alpha, a, lda, b, ldb, beta, ldc)
!
! l = rows in C
! n = number of cols in C
! m = cols in A (if 'n')
! alpha = scalar alpha
! a is the matrix A - A(l,m)
! lda = leading dimension of A (lda >= n)
! b is the matrix B - B(m,n)
! ldb leading dimension of b (ldb >= m)
! beta = scalar beta
! c =C(l,n)
! ldc = leading dimension of c

      call sgemm('n','n', m, n, k, 1.0, A, m, B, k, 0.0, D, m)

!      D = D_d

      return
   end subroutine
end module

