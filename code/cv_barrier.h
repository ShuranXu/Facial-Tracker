#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <atomic>
#include <stdexcept>

class CVBarrier
{
public:

    CVBarrier(std::size_t nb_threads)
        : m_mutex(),
        m_condition(),
        counter(0),
        instance(0),
        thread_num(nb_threads)
    {
        if (nb_threads == 0) {
            throw std::invalid_argument("Barrier thread count cannot be 0");
        }
    }

    CVBarrier(const CVBarrier& barrier) = delete;
    CVBarrier(CVBarrier&& barrier) = delete;
    ~CVBarrier() noexcept
    {
    }

    CVBarrier& operator=(const CVBarrier& barrier) = delete;
    CVBarrier& operator=(CVBarrier&& barrier) = delete;
    
    void Wait()
	{
		std::unique_lock< std::mutex > lock(m_mutex);
        std::size_t inst = instance; // store current instance for comparison
                                     // in predicate

		if (++counter == thread_num)
		{   
            std::cout << "notify" << std::endl;
            counter = 0;
            instance++; // increment instance for next use of barrier and to
                        // pass condition variable predicate
			m_condition.notify_all();
		}
		else
		{   
            std::cout << "wait" << std::endl;
            m_condition.wait(lock, [this, &inst]() { return instance != inst; });
		}
	}

private:

    std::mutex m_mutex;
    std::size_t instance; // counter to keep track of barrier use count
    std::condition_variable m_condition;
    std::size_t thread_num;
    std::atomic_int counter;
};
