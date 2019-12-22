#include "H5Tmodule.h"          /* This source code file is part of the H5T module */
#include "H5private.h"        /* Generic Functions            */

#include "H5Eprivate.h"        /* Error handling              */
#include "H5FLprivate.h"    /* Free Lists                */
#include "H5Iprivate.h"        /* IDs                      */
#include "H5Tpkg.h"        /* Datatypes                 */

/****************/
/* Local Macros */
/****************/

/******************/
/* Local Typedefs */
/******************/

/********************/
/* Package Typedefs */
/********************/

/********************/
/* Local Prototypes */
/********************/

/********************/
/* Public Variables */
/********************/

/*****************************/
/* Library Private Variables */
/*****************************/

/*********************/
/* Package Variables */
/*********************/

/*******************/
/* Local Variables */
/*******************/

/*-------------------------------------------------------------------------
 * Function:    H5T__init_native
 *
 * Purpose:    Initialize pre-defined native datatypes from code generated
 *              during the library configuration by H5detect.
 *
 * Return:    Success:    non-negative
 *        Failure:    negative
 *
 * Programmer:    Robb Matzke
 *              Wednesday, December 16, 1998
 *
 *-------------------------------------------------------------------------
 */
herr_t H5T__init_native(void) {
	H5T_t *dt = NULL;
	herr_t ret_value = SUCCEED;

	FUNC_ENTER_PACKAGE

		/*
		 *    0
		 * IIIIIIII
		 * Alignment: none
		 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_SCHAR_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_SCHAR_ALIGN_g = 1;
			H5T_NATIVE_SCHAR_COMP_ALIGN_g = 1;

			/*
			 *    0
			 * UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UCHAR_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UCHAR_ALIGN_g = 1;

			/*
			 *    1   0
			 * IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_SHORT_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_SHORT_ALIGN_g = 1;
			H5T_NATIVE_SHORT_COMP_ALIGN_g = 2;

			/*
			 *    1   0
			 * UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_USHORT_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_USHORT_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_g = H5I_register(H5I_DATATYPE, dt, FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_ALIGN_g = 1;
			H5T_NATIVE_INT_COMP_ALIGN_g = 4;

			/*
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_g = H5I_register(H5I_DATATYPE, dt, FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_LONG_g = H5I_register(H5I_DATATYPE, dt, FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_LONG_ALIGN_g = 1;
			H5T_NATIVE_LONG_COMP_ALIGN_g = 4;

			/*
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_ULONG_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_ULONG_ALIGN_g = 1;

			/*
			 *    0
			 * IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT8_g = H5I_register(H5I_DATATYPE, dt, FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT8_ALIGN_g = 1;

			/*
			 *    0
			 * UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT8_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT8_ALIGN_g = 1;

			/*
			 *    0
			 * IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_LEAST8_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_LEAST8_ALIGN_g = 1;

			/*
			 *    0
			 * UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_LEAST8_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_LEAST8_ALIGN_g = 1;

			/*
			 *    0
			 * IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_FAST8_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_FAST8_ALIGN_g = 1;

			/*
			 *    0
			 * UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 1;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 8;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_FAST8_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_FAST8_ALIGN_g = 1;

			/*
			 *    1   0
			 * IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT16_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT16_ALIGN_g = 1;

			/*
			 *    1   0
			 * UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT16_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT16_ALIGN_g = 1;

			/*
			 *    1   0
			 * IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_LEAST16_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_LEAST16_ALIGN_g = 1;

			/*
			 *    1   0
			 * UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_LEAST16_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_LEAST16_ALIGN_g = 1;

			/*
			 *    1   0
			 * IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_FAST16_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_FAST16_ALIGN_g = 1;

			/*
			 *    1   0
			 * UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 2;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 16;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_FAST16_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_FAST16_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT32_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT32_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT32_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT32_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_LEAST32_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_LEAST32_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_LEAST32_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_LEAST32_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_FAST32_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_FAST32_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_FAST32_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_FAST32_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT64_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT64_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_LEAST64_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_LEAST64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_LEAST64_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_LEAST64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_INT_FAST64_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_INT_FAST64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_UINT_FAST64_g = H5I_register(H5I_DATATYPE, dt,
			FALSE)) < 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_UINT_FAST64_ALIGN_g = 1;

			/*
			 *    7   6   5   4
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 *    3   2   1   0
			 * IIIIIIII IIIIIIII IIIIIIII IIIIIIII
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_2;
			if ((H5T_NATIVE_LLONG_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_LLONG_ALIGN_g = 1;
			H5T_NATIVE_LLONG_COMP_ALIGN_g = 8;

			/*
			 *    7   6   5   4
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 *    3   2   1   0
			 * UUUUUUUU UUUUUUUU UUUUUUUU UUUUUUUU
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_INTEGER;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.i.sign = H5T_SGN_NONE;
			if ((H5T_NATIVE_ULLONG_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_ULLONG_ALIGN_g = 1;

			/*
			 *    3   2   1   0
			 * SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
			 * Implicit bit? yes
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_FLOAT;
			dt->shared->size = 4;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 32;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.f.sign = 31;
			dt->shared->u.atomic.u.f.epos = 23;
			dt->shared->u.atomic.u.f.esize = 8;
			dt->shared->u.atomic.u.f.ebias = 0x0000007f;
			dt->shared->u.atomic.u.f.mpos = 0;
			dt->shared->u.atomic.u.f.msize = 23;
			dt->shared->u.atomic.u.f.norm = H5T_NORM_IMPLIED;
			dt->shared->u.atomic.u.f.pad = H5T_PAD_ZERO;
			if ((H5T_NATIVE_FLOAT_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_FLOAT_ALIGN_g = 1;
			H5T_NATIVE_FLOAT_COMP_ALIGN_g = 4;

			/*
			 *    7   6   5   4
			 * SEEEEEEE EEEEMMMM MMMMMMMM MMMMMMMM
			 *    3   2   1   0
			 * MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
			 * Implicit bit? yes
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_FLOAT;
			dt->shared->size = 8;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 64;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.f.sign = 63;
			dt->shared->u.atomic.u.f.epos = 52;
			dt->shared->u.atomic.u.f.esize = 11;
			dt->shared->u.atomic.u.f.ebias = 0x000003ff;
			dt->shared->u.atomic.u.f.mpos = 0;
			dt->shared->u.atomic.u.f.msize = 52;
			dt->shared->u.atomic.u.f.norm = H5T_NORM_IMPLIED;
			dt->shared->u.atomic.u.f.pad = H5T_PAD_ZERO;
			if ((H5T_NATIVE_DOUBLE_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_DOUBLE_ALIGN_g = 1;
			H5T_NATIVE_DOUBLE_COMP_ALIGN_g = 8;

			/*
			 *   15  14  13  12
			 * ???????? ???????? ???????? ????????
			 *   11  10   9   8
			 * ???????? ???????? SEEEEEEE EEEEEEEE
			 *    7   6   5   4
			 * MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
			 *    3   2   1   0
			 * MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
			 * Implicit bit? no
			 * Alignment: none
			 */
			if (NULL == (dt = H5T__alloc()))
				HGOTO_ERROR(H5E_DATATYPE, H5E_NOSPACE, FAIL,
						"datatype allocation failed")
			dt->shared->state = H5T_STATE_IMMUTABLE;
			dt->shared->type = H5T_FLOAT;
			dt->shared->size = 16;
			dt->shared->u.atomic.order = H5T_ORDER_LE;
			dt->shared->u.atomic.offset = 0;
			dt->shared->u.atomic.prec = 80;
			dt->shared->u.atomic.lsb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.msb_pad = H5T_PAD_ZERO;
			dt->shared->u.atomic.u.f.sign = 79;
			dt->shared->u.atomic.u.f.epos = 64;
			dt->shared->u.atomic.u.f.esize = 15;
			dt->shared->u.atomic.u.f.ebias = 0x00003fff;
			dt->shared->u.atomic.u.f.mpos = 0;
			dt->shared->u.atomic.u.f.msize = 64;
			dt->shared->u.atomic.u.f.norm = H5T_NORM_NONE;
			dt->shared->u.atomic.u.f.pad = H5T_PAD_ZERO;
			if ((H5T_NATIVE_LDOUBLE_g = H5I_register(H5I_DATATYPE, dt, FALSE))
					< 0)
				HGOTO_ERROR(H5E_DATATYPE, H5E_CANTINIT, FAIL,
						"can't register ID for built-in datatype")
			H5T_NATIVE_LDOUBLE_ALIGN_g = 1;
			H5T_NATIVE_LDOUBLE_COMP_ALIGN_g = 16;

			/* Set the native order for this machine */
			H5T_native_order_g = H5T_ORDER_LE;

			/* Structure alignment for pointers, hvl_t, hobj_ref_t, hdset_reg_ref_t */
			H5T_POINTER_COMP_ALIGN_g = 8;
			H5T_HVL_COMP_ALIGN_g = 8;
			H5T_HOBJREF_COMP_ALIGN_g = 8;
			H5T_HDSETREGREF_COMP_ALIGN_g = 1;

			done: if (ret_value < 0) {
				if (dt != NULL) {
					dt->shared = H5FL_FREE(H5T_shared_t, dt->shared);
					dt = H5FL_FREE(H5T_t, dt);
				} /* end if */
			} /* end if */

			FUNC_LEAVE_NOAPI(ret_value);
} /* end H5T__init_native() */

/****************************************/
/* ALIGNMENT and signal-handling status */
/****************************************/
/* Signal() support: yes */
/* setjmp() support: yes */
/* longjmp() support: yes */
/* sigsetjmp() support: no */
/* siglongjmp() support: no */
/* sigprocmask() support: no */

/******************************/
/* signal handlers statistics */
/******************************/
/* signal_handlers tested: 15 times */
/* sigbus_handler called: 5 times */
/* sigsegv_handler called: 5 times */
/* sigill_handler called: 5 times */

