// a C++ implementation of a KD Tree
#include <stddef.h>
#include <iostream>

using namespace std;

// currently supports 2 dimensions
const int k = 2;

// a structure to represent node of kd tree
struct Node 
{
    int point[k]; // to store k dimensional point
    Node *left, *right;
};

// a method to create a node of kd tree
struct Node* newNode(int arr[]) 
{
    
    struct Node* temp = new Node;

    for (int i = 0; i < k; i++) 
        temp->point[i] = arr[i];
    
    temp->left = temp->right = NULL;
    return temp;
}

// inserts a new node and returns root of modified tree
Node* insertRec(Node* root, int point[], unsigned depth) 
{
    // base case
    if (root == NULL) 
        return newNode(point);
    
    // calculate current dimension (cd) of comparison
    unsigned cd = depth % k;

    // compare the new point with root on current dimension and decide left or right subtree
    if (point[cd] < root->point[cd])
        root->left = insertRec(root->left, point, depth + 1);
    else
        root->right = insertRec(root->right, point, depth + 1);
    
    return root;
}

// function to insert a new point with given point in kd tree and return new root
Node* insert(Node* root, int point[])
{
    return insertRec(root, point, 0);
}

// a utility method to determine if two points are the same in k dimensional space
bool arePointsSame(int point1[], int point2[])
{
    for (int i = 0; i < k; i++)
        if (point1[i] != point2[i])
            return false;
    
    return true;
}

// searches for a point in the kd tree
bool searchRec(Node* root, int point [], unsigned depth)
{
    // base case
    if (root == NULL)
        return false;
    if (arePointsSame(root->point, point))
        return true;

    unsigned cd = depth % k;

    if (point[cd] < root->point[cd])
        return searchRec(root->left, point, depth + 1);
    
    return searchRec(root->right, point, depth + 1);
}

// searches for a point in the kd tree
bool search(Node* root, int point[])
{
    return searchRec(root, point, 0);
}
 // driver program to test above functions
int main() 
{
    struct Node* root = NULL;
    int points[][k] = {{3, 6}, {17, 15}, {13, 15}, {6, 12},
                        {9, 1}, {2, 7}, {10, 19}};
    
    int n = sizeof(points)/sizeof(points[0]);

    for (int i = 0; i < n; i++)
        root = insert(root, points[i]);

    int point1[] = {10, 19};
    (search(root, point1)) ? cout << "Found\n" : cout << "Not Found\n";

    int point2[] = {12, 19};
    (search(root, point2)) ? cout << "Found\n" : cout << "Not Found\n";

    return 0;
}




























