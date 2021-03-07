#pragma once

#ifndef HV_RAII
#define HV_RAII

#include <functional>

namespace hv::v1 {
	class raii {
	private:
		std::function<void()> instance;
	public:

		raii(std::function<void()> _instance) : instance(_instance) {

		}
		~raii() {
			instance();
		}
	};
}


#endif // !HV_RAII
