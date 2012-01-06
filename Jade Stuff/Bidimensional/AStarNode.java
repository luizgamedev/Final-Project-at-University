/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package TestBidimensional;

/**
 *
 * @author kazenotenshi
 */
public class AStarNode {

    public int F;
    public int G;
    public int H;
    public Point thisPoint;
    public AStarNode father;

    public AStarNode(Point p){
        thisPoint = new Point();
        thisPoint.x = p.x;
        thisPoint.y = p.y;
        this.father = null;
    }

    public AStarNode(Point p, Point father){
        thisPoint = p;
        this.father.thisPoint = father;
    }

}
