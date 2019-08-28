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
module laplace2d
  public :: initialize
  public :: calcNext
  public :: swap
  public :: dealloc
  contains
    subroutine initialize(A, Anew, m, n)
      integer, parameter :: fp_kind=kind(1.0d0)
      real(fp_kind),allocatable,intent(out)   :: A(:,:)
      real(fp_kind),allocatable,intent(out)   :: Anew(:,:)
      integer,intent(in)          :: m, n

      allocate ( A(0:n-1,0:m-1), Anew(0:n-1,0:m-1) )

      A    = 0.0_fp_kind
      Anew = 0.0_fp_kind

      A(0,:)    = 1.0_fp_kind
      Anew(0,:) = 1.0_fp_kind
    end subroutine initialize

    function calcNext(A, Anew, m, n)
      integer, parameter          :: fp_kind=kind(1.0d0)
      real(fp_kind),intent(inout) :: A(0:n-1,0:m-1)
      real(fp_kind),intent(inout) :: Anew(0:n-1,0:m-1)
      integer,intent(in)          :: m, n
      integer                     :: i, j
      real(fp_kind)               :: error

      error=0.0_fp_kind
      !$acc kernels present(A,Anew)
      !$acc loop gang worker(4)
      do j=1,m-2
        !$acc loop vector(32)
        do i=1,n-2
          Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
                                       A(i  ,j-1) + A(i  ,j+1) )
          error = max( error, abs(Anew(i,j)-A(i,j)) )
        end do
      end do
      !$acc end kernels
      calcNext = error
    end function calcNext

    subroutine swap(A, Anew, m, n)
      integer, parameter        :: fp_kind=kind(1.0d0)
      real(fp_kind),intent(out) :: A(0:n-1,0:m-1)
      real(fp_kind),intent(in)  :: Anew(0:n-1,0:m-1)
      integer,intent(in)        :: m, n
      integer                   :: i, j

      !$acc kernels present(A,Anew)
      !$acc loop gang worker(4)
      do j=1,m-2
        !$acc loop vector(32)
        do i=1,n-2
          A(i,j) = Anew(i,j)
        end do
      end do
      !$acc end kernels
    end subroutine swap

    subroutine dealloc(A, Anew)
      integer, parameter :: fp_kind=kind(1.0d0)
      real(fp_kind),allocatable,intent(in) :: A
      real(fp_kind),allocatable,intent(in) :: Anew

      deallocate (A,Anew)
    end subroutine
end module laplace2d
