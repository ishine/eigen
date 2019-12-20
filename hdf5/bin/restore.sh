#!/bin/sh
#
# Copyright by The HDF Group.                                              
# All rights reserved.                                                     
#                                                                          
# This file is part of HDF5. The full HDF5 copyright notice, including     
# terms governing use, modification, and redistribution, is contained in   
# the COPYING file, which can be found at the root of the source code
# distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.
# If you do not have access to either file, you may request a copy from
# help@hdfgroup.org.
#

# A script to clean up the action of autogen.sh
#
# If this script fails to clean up generated files on a particular
# platform, please contact help@hdfgroup.org or comment on the forum.

echo
echo "*******************************"
echo "* HDF5 autogen.sh undo script *"
echo "*******************************"
echo

echo "Remove autom4te.cache directory"
rm -rf autom4te.cache

echo "Remove configure script"
rm -f configure

echo "Remove Makefile.in files"
find . -type f -name 'Makefile.in' -exec rm {} \;

echo "Remove files generated by libtoolize"
rm -f bin/ltmain.sh
rm -f m4/libtool.m4
rm -f m4/ltoptions.m4
rm -f m4/ltsugar.m4
rm -f m4/ltversion.m4
rm -f m4/lt~obsolete.m4

echo "Remove files generated by automake"
rm -f bin/compile
rm -f bin/config.guess
rm -f bin/config.sub
rm -f bin/install-sh
rm -f bin/missing
rm -f bin/test-driver
rm -f bin/depcomp

echo "Remove files generated by bin/make_err"
rm -f src/H5Epubgen.h
rm -f src/H5Einit.h
rm -f src/H5Eterm.h
rm -f src/H5Edefin.h

echo "Remove files generated by bin/make_vers"
rm -f src/H5version.h

echo "Remove files generated by bin/make_overflow"
rm -f src/H5overflow.h

echo "Remove remaining generated files"
rm -f aclocal.m4

