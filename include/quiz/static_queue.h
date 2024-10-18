#include <array>
#include <iostream>

template <typename T, std::size_t Size>
class StaticQueue {
public:
    StaticQueue() : front(0), rear(0), count(0) {}

    bool push(const T& item) {
        if (full()) {
            std::cerr << "Queue is full\n";
            return false;
        }
        data[rear] = item;
        rear = (rear + 1) % Size;
        count++;
        return true;
    }

    bool pop(T& item) {
        if (empty()) {
            std::cerr << "Queue is empty\n";
            return false;
        }
        item = data[front];
        front = (front + 1) % Size;
        count--;
        return true;
    }

    bool empty() const {
        return count == 0;
    }

    bool full() const {
        return count == Size;
    }

    std::size_t size() const {
        return count;
    }

private:
    std::array<T, Size> data;
    std::size_t front;
    std::size_t rear;
    std::size_t count;
};