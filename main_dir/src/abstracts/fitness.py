import os, time

def timer_write(ga_name, start, finish):
    timer_out_cpu = open(os.getcwd()+ "\\timer_directory\\cpuaddadversarial11outreal.txt", "a+")
    timer_out_cpu.write(ga_name + " CPU time is " + str(finish - start) + "\n")
    timer_out_cpu.close()

# Time complexity = O(mxn)
def lc_substring(X, Y, m, n):
    # LCSuff is the table with zero  
    # value initially in each cell 
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
      
    # To store the length of  
    # longest common substring 
    result = 0 
    start = time.perf_counter()
    # Following steps to build 
    # LCSuff[m+1][n+1] in bottom up fashion 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if (i == 0 or j == 0): 
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]): 
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j]) 
            else: 
                LCSuff[i][j] = 0
    finish = time.perf_counter()
    timer_write("Evaluate", start, finish)
    return result

# Time complexity: O(3^m) naive approach
def edit_distance(str1, str2, m, n): 
    if m == 0:
        return n

    if n == 0:
        return m

    if str1[m - 1] == str2[n - 1]:
        return edit_distance(str1, str2, m - 1, n - 1)

    return 1 + min(edit_distance(str1, str2, m, n - 1), edit_distance(str1, str2, m - 1, n), edit_distance(str1, str2, m - 1, n - 1))

# Time complexity: O(mxn) #insert point 1, delete point 1, replace point 0 or 1
def edit_distance3(str1, str2, m, n): 
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m+1): 
        for j in range(n+1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])      # Replace 
  
    return dp[m][n]

# Time complexity: O(mxn) #insert point 1, delete point 0, replace point 0 or 1
def edit_distance2(str1, str2, m, n): 
    #print("str 1 is ", str1, "str2 is ", str2, " ",m, " ", n)
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
    start = time.perf_counter()
    # Fill d[][] in bottom up manner 
    for i in range(0, m+1): 
        for j in range(0, n+1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string  (delt = 1)
            
            # If last character are different, consider all 
            # possibilities and find minimum (delt = 0)
            else: 
                delt = 1
                if str1[i - 1] != str2[j - 1]: 
                    delt = 0 
                dp[i][j] = min(dp[i][j-1] + 1,        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1] + delt)      # Replace 
  
    #print("dp is ", dp)
    finish = time.perf_counter()
    timer_write("Evaluate", start, finish)
    return dp[m][n]

# Time complexity: O(n) where n is the number of different characters
def hamming_distance(s1, s2):
    # Return the Hamming distance between equal-length sequences
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

# Time complexity: O(nm) where n and m is the length of the two strings
def lc_subsequence(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in range(m+1)] 
    start = time.perf_counter()
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    start = time.perf_counter()
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    finish = time.perf_counter()
    timer_write("Evaluate", start, finish)
    return L[m][n]