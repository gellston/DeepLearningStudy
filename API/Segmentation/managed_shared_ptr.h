#pragma once

#include <memory>


namespace HV {

    namespace V1 {
        template <class T>
        public ref class mananged_shared_ptr sealed
        {
        private:
            std::shared_ptr<T>* pPtr;

        public:


            mananged_shared_ptr()
                : pPtr(new std::shared_ptr<T>())
            {}

            mananged_shared_ptr(T* t) {
                pPtr = new std::shared_ptr<T>(t);
            }

            //mananged_shared_ptr(std::shared_ptr<T> t) {
            //    pPtr = new std::shared_ptr<T>(t);
            //}

            mananged_shared_ptr(const mananged_shared_ptr<T>% t) {
                pPtr = new std::shared_ptr<T>(*t.pPtr);
            }

            mananged_shared_ptr(std::shared_ptr<T>& t) {
                pPtr = new std::shared_ptr<T>(t);
            }

            !mananged_shared_ptr() {
                if (pPtr != nullptr) {
                    delete pPtr;
                    pPtr = nullptr;
                }

            }

            ~mananged_shared_ptr() {
                if (pPtr != nullptr) {
                    delete pPtr;
                    pPtr = nullptr;
                }

            }

            operator std::shared_ptr<T>() {
                return *pPtr;
            }

            mananged_shared_ptr<T>% operator=(T* ptr) {
                delete pPtr;
                pPtr = new std::shared_ptr<T>(ptr);
                return *this;
            }

            void operator=(std::shared_ptr<T>& ptr) {
                *pPtr = ptr;
            }

            T* operator->() {
                return (*pPtr).get();
            }

            void reset() {
                pPtr->reset();
            }



            std::shared_ptr<T>& get() {
                return *pPtr;
            }
        };
    }
}