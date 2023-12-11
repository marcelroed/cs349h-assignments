### Part A: Hyperdimensional Computing [8 pts total, 2 pts / question]

**Task 1:** Implement the hypervector generation, binding, bundling, hamming distance, and permutation operations in the scaffold code, and then implement an HD encoding procedure that encodes strings. Also, implement the `add` and `get` functions in the item memory memory class; these functions will be used to manage the atomic hypervectors. 

For example "fox" would be translated to sequence ["f","o", "x"]. For simplicity, use a hypervector size of 10,000 to answer these questions unless otherwise stated.

**Q1.** Construct a HDC-based string for the word "fox". How did you encode the "fox" string? How similar is the hypervector for "fox" to the hypervector for "box" in your encoding? How similar is the hypervector for "xfo"? How similar is the hypervector for "car"? Please remark on the relative similarities, not the absolute distances.

__A1.__
I encoded the string "fox" by permuting each letter by its position in the string, starting at 0, then bundled the resulting hypervectors together.
Sanity checking the similarity between "fox" and "fox" gives us 0.0, as expected.
When comparing "fox" and "box" we get a relatively small distance of ~0.25, while both "xfo" and "car" have distances of ~0.5, which is as expected when the hypervectors are randomly generated.


**Q2.** Change your encoding so the order of the letters doesn't matter. What changes did you make? Please remark on the relative similarities, not the absolute distances.

__A2.__
By removing the permutation step, the order of the letters no longer matters.
We see, as expected, that now the distance between "fox" and "xfo" is 0.
The distance between "fox" and "box" is small at ~0.26, and the distance between "fox" and "car" is still ~0.5.

-------


**Task 2**: Implement the bit flip error helper function (`apply_bit_flips`). Then apply bit flip errors to hypervectors before they are stored in item memory, where the bit flip probability is 0.01. Use the `monte_carlo`, `study_distributions`, and `plot_hist_distributions` helper functions study the distribution of distances between `fox` and `box`, compared to the distance between `fox` and `car` with and without hardware error.

**Q3.** Try modifying the hardware error rate (`perr`). How high can you make the hardware error the two distributions begin to become indistinguishable? What does it mean conceptually when the two distance distributions have a lot of overlap?

__A3.__
As the hardware error rate increases, the distributions get closer to each other, and they start overlapping.
Until we get to a very high error rate of 0.98 and higher, the distributions are still very easy to tell apart, however, for a given sample it might be hard to tell which distribution it came from, even at a lower error rate of around 0.8.
Still, this is exceptional robustness to hardware error.

**Q4.** Try modifying the hypervector size (`SIZE`). How small can you make the word hypervectors before the two distributions begin to become indistinguishable? 

__A4.__
Again, we can decrease the size of the hypervectors dramatically to get to a size of closer to 2 or 3 before the distributions become indistinguishable at a trial size of 1000 with a hardware error rate of 0.10 as set originally.
This robustness is likely mostly due to running a high number of samples, which means that the distributions are more likely to betray the statistical differences between the vectors even so much noise.

-----

**Task 3**: Next, fill out the stubs in the item memory class -- there are stubs for threshold-based and winner-take-all queries, and for computing the hamming distances between item memory rows and the query hypervector. The item memory class will be used in later exercises to build a database data structure and an ML model. 


### Part B: Item Memories [`hdc-db.py`, 10 points total, 2 pts / question]

Next, we will use this item memory to implement a database data structure; we will be performing queries against an HDC-based database populated with the digimon dataset (`digimon.csv`).  The HDDatabase class implements the hyperdimensional computing-based version of this data structure, and contains stubs of convenience functions for encoding / decoding strings and database rows, as well as stubs for populating and querying the database. We will implement this class and then invoke `build_digimon_database` to build a database using the HDDatabase class. For simplicity, use a hypervector size of 10,000 to answer these questions unless otherwise stated.

_Tip_: For this exercise, map every string to an atomic hypervector. This will keep retrieval tasks relatively simple. For decoding operations, you will likely need to use the self-inverse property of binding and perform additional lookups to recover information.

---------

__Task 0__: The database data structure contains multiple rows, where each row is uniquely identified with a primary key and contains a collection of fields that are assigned to values. In the HD implmentation, the database rows are maintained in item memories, and row data is encoded as a hypervector. Decide how you want to map the database information to item memories. Implement the database row addition `add_row` function, which should invoke the `encode_row` helper function and update the appropriate item memories.

---------

__Task 1__: Implement the string and row encoding functions (`encode_string`, `encode_row`). These encoding functions accept a string and a database row (field-value map) respectively, and translate these inputs into hypervectors by applying HD operators to atomic basis vectors. Then, implement the string and row decoding functions (`decode_string`, `decode_row`) which take the hypervector representations of a string or database row respectively and reconstructs the original data. The decoding routines will likely need to perform multiple codebook / item memory lookups and use HD operator properties (e.g., unbinding) to recover the input data. Execute `digimon_test_encoding` function to test your string and row encoding routines and verify that you're able to recover information from the hypervector embedding with acceptable reliability. 

**Q1.** Describe how you encoded strings / database rows as hypervectors. Write out the HD expression you used to encode each piece of information, and describe any atomic hypervectors you introduced.

__A1.__
Strings are encoded as hypervectors by generating atomic hypervectors for each string individually.
This is done separately for each kind of string, so we split codebooks for primary keys, fields, and values.
The rows are encoded by binding each field with its value, then bundling together all the resulting hypervectors.
This makes it easy to later decode any part of the row by unbinding and doing distance comparisons either with winner-takes-all or threshold-based queries.
Encoding strings is done by just generating a random hypervector.
The HD expression for encoding a row is:

$$
\mathrm{hv}(\mathrm{row}) = \sum_f \mathrm{hv}(k_f) \odot \mathrm{hv}(v_f).
$$


**Q2.** Describe how you decoded the strings / database rows from hypervectors. Describe any HD operations you used to isolate the desired piece of information, and describe what item memory lookups you performed to recover information. If you're taking advantage of any HD operator properties to isolate information, describe how they do so.

__A2.__
In order to decode strings, we do direct comparisons with the hypervectors in each codebook.
We can do this either by winner-takes-all or threshold-based queries.
If we have a hyperdimensional vector $\mathrm{hv}_s$, we can decode it by computing all the distances
$$
\langle \mathrm{hv}_s, \mathrm{hv}_{s_i} \rangle\  \forall s_i \in \mathrm{codebook}
$$
and choosing the smallest one (WTA).
To decode a database row, we iterate through the all field hypervectors in the field codebook, bind them with the row hypervector, and perform threshold-based queries on the resulting hypervectors to the value codebook.
If the lowest resulting distance is less than a certain threshold, we pick this value from the codebook.
Recall that the expression for the row hypervector is
$$
\mathrm{hv}(\mathrm{row}) = \sum_f \mathrm{hv}(k_f) \odot \mathrm{hv}(v_f).
$$
Thus binding with the hypervector for key $k_l$ gives
$$
\begin{align*}
\mathrm{hv}(k_l) \odot \mathrm{hv}(\mathrm{row}) &= \mathrm{hv}(k_l) \sum_f \mathrm{hv}(k_f) \odot \mathrm{hv}(v_f) \\
&= \sum_f \mathrm{hv}(k_l)\odot\mathrm{hv}(k_f) \odot \mathrm{hv}(v_f) \\
&= \sum_{f \neq l} \mathrm{hv}(k_l)\odot\mathrm{hv}(k_f) \odot \mathrm{hv}(v_f)\\ &\ \quad\ \  + \mathrm{hv}(k_l)\odot\mathrm{hv}(k_l) \odot \mathrm{hv}(v_l) \\
&= \sum_{f \neq l} \mathrm{hv}(k_l)\odot\mathrm{hv}(k_f) \odot \mathrm{hv}(v_f)\\ &\ \quad\ \  + \mathrm{hv}(v_l), \\
\end{align*}
$$
where we use that binding distributes over bundling and that binding with the same vector is self-inverting.
Recall from lecture that a binding of vectors is similar to all of the vectors, and thus its distance to those vectors is low.
When we now compare this hypervector to the value codebook, we get a high distance for all values except for the one that was bound to the key $k_l$, represented by $v_l$.

--------



__Task 2__: Next, we'll implement routines for querying the data structure. Implement the `get_value` and `get_matches` stubs -- the `get_value` query retrieves the value assigned to a user-provided field within a record. The `get_matches` stub retrieves the rows that contain a subset of field-value pairs. Implement both these querying routines and then execute `digimon_basic_queries` and `digimon_value_queries` to test your implementations.

**Q3.** How did you implement the `get_value` query? Describe any HD operators and lookups you performed to implement this query.

__A3.__ 
`get_value` is implemented by performing a simplification of the decoding operation used for the row above.
Given the field $k_f$ and the row hypervector $\mathrm{hv}(\mathrm{row})$, we compute the value for the given field by binding with the field hypervector and performing a threshold-based query on the resulting hypervector with the value codebook.
We are thus using the exact same math as above, but for only one value of $l$, namely the one corresponding to the field we are extracting.

**Q4.** How did you implement the `get_matches` query? Describe any HD operators and lookups you performed to implement this query. How high of a distance threshold can you set before you start seeing false positives in the returned results. 

__A4.__
`get_matches` is done by encoding the dictionary provided, as explained previously, and then performing a simple threshold-based query between this vector and all the vectors in the row database.
The result is a dictionary of all the primary keys that match the query along with their distances to the hypervector encoded from the query dictionary.
We will find entries in the database where the encoded row is similar to the encoded query, which is what we want.

-----

__Task 3__: Implement the `get_analogy` query, which given two records, identifies the value that shares the same field as the value supplied in the query. For example, if you perform an `analogy` query on the `US` and `Mexico` records, and ask for the value in the `Mexico` record that relates to the `Dollar` value in the `US` record, this query would return `Peso`. This query completes the analogy  _Dollar is to USA as <result> is to Mexico_.

_Tip_: If you want more information on this type of query, you can look up "dollar value of mexico"

**Q5.** How did you implement the `get_analogy` query? Describe how this is implemented using HD operators and item memory lookups. Why does your implementation work? You may want to walk through and HD operator properties you leveraged to complete this query.

__A5.__
The `get_analogy` query collects the target hypervector from the row database, and binds it to the hypervector of `target_value` (fetched from the value codebook).
We compare the resulting hypervector to all fields and select the best using WTA.
This is the key that we want to look up in the other row.
We decode the field from the other row as described earlier, which is then returned along with a distance measure.



### Part C: Implementing the HDC Classifier [10 pts total, 2 pts/question, hdc-ml.py]

Next, we will use an item memory to implement an MNIST image classifier. A naive implementation of this classifier should easily be able to get ~75% accuracy. In literature, HDC-based MNIST classifiers have been shown to achieve ~95% classification accuracy. In this exercise, you will implement the necessary encoding/decoding routines, and you will implement both the training and inference algorithms for the classifier. 

__Tips__: Try a simple pixel/image encoding first.  For decoding operations, you will likely need to use the self-inverse property of binding to recover information.

-------------

**Task 1**: Fill in the `encode_pixel`, `decode_pixel`, `encode_image`, and `decode_image` stubs in the MNIST classifier. These functions should translate pixels/images to/from their hypervector representation. Then use the `test_encoding` function to evaluate the quality of your encoding. This function will save the original image to `sample0.png`, and the encoded then decoded image to `sample0_rec.png`.

**Q1.** How did you encode pixels as a hypervector? Write out the HD expressions, and describe what atomic/basis hypervectors you used for both encodings. 

__A1.__
Pixels are encoded as a hypervector by binding together the indices of each pixel (with the y-coordinate permuted by 1) together with the values of each pixel.
The value hypervectors are stored in a separate codebook from the indices.
The expression for each pixel is
$$
\mathrm{hv}(\mathrm{pixel}(i, j, v)) = \mathrm{hv}(i)\odot p_1(\mathrm{hv}(j)) \odot \mathrm{hv}(v)
$$


**Q2.** How did you encode images as a hypervector? Write out the HD expressions, and describe any atomic/basis hypervectors in the expression. 

__A2.__
Each image is encoded as the bundle of all the pixels in the image:
$$
\sum_{i, j} \mathrm{hv}(\mathrm{pixel}(i, j, v)) = \sum_{i, j} \mathrm{hv}(i)\odot p_1(\mathrm{hv}(j)) \odot \mathrm{hv}(v).
$$
-----------------------

**Task 2**: Fill in the `train` and the `classify` stubs in the MNIST classifier. Test your classifier out by invoking the `test_classifier` function. What classification accuracy did you attain? 

__I got an accuracy of around 78%, but this varies from run to run.__

**Q3.** What happens to the classification accuracy when you reduce the hypervector size; how small of a size can you select before you see > 5% loss in accuracy? 

__A3.__
Reducing the hypervector size lower than 10 000 results in a loss of accuracy.
At a length of 2000 we see the accuracy drop by around 5%. 

**Q4.** What happens to the classification accuracy when you introduce bit flips into the item memory's distance calculation? How much error can you introduce before you see a > 5% loss in accuracy?

__A4.__
Introducing bit flips into the distance calculation by changing the error rate in the classifiers item memory also lowers our accuracy, but a very high error rate is needed to get a 5% loss in accuracy.
at roughly 29% error rate the accuracy drops by 5%.

------------------------

**Task 3**: You can also implement a generative model using hyperdimensional computing. In this following exercise, we will use HDC to generate images of handwritten digits from the classifier label. Naive HD generative models are very similar to HD classifiers, and are constructed in two simple steps:

- _Constructing a Generative Model._ For each classifier label, group training data by label, and translate each datum to a hypervector. Next, generate a probability vector for each label. The probability value at position i of the probability vector is the probability that the hypervector bit in position i is a "1" bit value. The probability vector can easily be computed by summing up the M hypervectors that share the same label, and then normalizing by 1/M.

- _Generating Random Images._ To generate a random image for some label, you sample a binary hypervector from the probability hypervector for that label. You then translate the hypervector to an image using the hypervector decoding routine (`decode_image`). You can sample and bundle multiple hypervectors to average the result.

**Q5.** Include a few pictures outputted by your generative model in your submission.

__A5.__
I have included some of the generated images in the `hw2-hyperdim-computing/` directory.
They can also be seen below if this markdown document is opened in the right directory.

![image1](generated_cat2_idx8.png)
![image2](generated_cat3_idx2.png)
![image3](generated_cat3_idx7.png)
![image4](generated_cat3_idx9.png)
![image5](generated_cat5_idx6.png)
![image6](generated_cat6_idx4.png)
![image6](generated_cat7_idx1.png)
![image6](generated_cat7_idx3.png)
![image6](generated_cat8_idx5.png)
![image6](generated_cat9_idx0.png)
