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
