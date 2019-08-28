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

module image
  use iso_c_binding
  interface 
    function readImage(path, width, height, nchannels) bind(C, name="readImage")
      use iso_c_binding
      character(kind=c_char) :: path(*)
      integer(kind=c_long) :: width, height, nchannels
      character(kind=c_char) :: readImage
    end function
  end interface
  interface 
    subroutine wrImage(path) bind(C, name="writeImage")
      use iso_c_binding
      character(kind=c_char) :: path(*)
    end subroutine
  end interface
end module
#ifdef DEBUG
program main
  use image
  integer(kind=c_long) :: w, h, c
  character(kind=c_char), pointer :: img(:)

  img = readImage(TRIM("avatar.jpg")//C_NULL_CHAR, w, h, c)

  print *, w, h, c

  !wrImage( TRIM("out_f.jpg") // C_NULL_CHAR )

end program
#endif
