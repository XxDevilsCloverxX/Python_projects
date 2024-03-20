import numpy as np

def generate_mini_batches(*data_lists, batch_size):
    num_lists = len(data_lists)
    assert all(len(data) == len(data_lists[0]) for data in data_lists), "All data lists must have the same length"
    
    combined_data = list(zip(*data_lists))
    np.random.shuffle(combined_data)
    
    num_samples = len(combined_data)
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        mini_batch = combined_data[start_index:end_index]
        batch_data = [np.array(batch) for batch in zip(*mini_batch)]
        yield tuple(batch_data)

    # Yield the remaining samples if the total number of samples is not divisible by batch_size
    if num_samples % batch_size != 0:
        mini_batch = combined_data[num_batches * batch_size:]
        batch_data = [np.array(batch) for batch in zip(*mini_batch)]
        yield tuple(batch_data)

# Example usage
list1 = np.arange(10)
list2 = np.arange(10, 20)
list3 = np.arange(20, 30)
list4 = np.arange(30, 40)
batch_size = 3

for mini_batch_data in generate_mini_batches(list1, list2, list3, list4, batch_size=batch_size):
    print("Mini-batch:")
    for i, data in enumerate(mini_batch_data):
        print(f"List {i+1}:", data)
