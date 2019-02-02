/*
 * umap_c.h
 *
 *  Created on: Feb 1, 2019
 *      Author: cjnolet
 */

#ifndef UMAP_C_H_
#define UMAP_C_H_


namespace ML {

	class UMAP {

	public:

		template<typename T>
		void fit(T *X, T *y = nullptr);

		template<typename T>
		void transform(T *X);
	};

}



#endif /* UMAP_C_H_ */
