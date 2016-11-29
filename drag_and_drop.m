function [a, properties] = drag_and_drop(x, y, r, color)
d = r*2;
px = x;
py = y;
dragging = [];
orPos = [];

set(gcf,'WindowButtonUpFcn',@dropObject,'units','normalized','WindowButtonMotionFcn',@moveObject);

a = rectangle('Position',[px py d d],'Curvature',[1,1],...
    'ButtonDownFcn',@dragObject,'EdgeColor', color, 'LineWidth', 2);


properties = get(a, 'Position');

    function dragObject(hObject,eventdata)
        selection_type = get(gcf, 'SelectionType');
        dragging = hObject;
        orPos = get(a,'Position');
        face_color = get(a, 'FaceColor');
        if strcmp(selection_type, 'open')
            if isempty(face_color) || strcmp('none', face_color)
                set(a, 'FaceColor', color)                
            else
                set(a, 'FaceColor', 'None')               
            end
        end
        
    end
    function dropObject(hObject,eventdata)
        
        dragging = [];
        
    end
    function moveObject(hObject,eventdata)
        selection_type = get(gcf,'SelectionType');
        newPos = get(gca,'CurrentPoint');
        %current_position = get(a, 'Position');        
        if ~isempty(dragging)
            if ~strcmp('extend', selection_type)
                current_position = get(a, 'Position');
                %Avoid to the circle to flip to the right.
                posDiff(1) = newPos(1, 1) - orPos(1, 1);
                posDiff(2) = -(newPos(1, 2) - orPos(1, 2));
                current_position = current_position +...
                    [[posDiff(1) -posDiff(2)] 0 0];
                if d > 0
                    %current_position(3:4) = d;
                end
                orPos = newPos(1, 1:2);
                set(dragging,'Position',current_position);
            else
                rectangle_position = get(a, 'Position');
                mouse_position = get(gca,'CurrentPoint');
                new_radius = (mouse_position(1) - rectangle_position(1));
                new_rectangle_position = [rectangle_position(1:2) new_radius new_radius];
                if all(new_rectangle_position > 0)
                    set(a, 'Position', new_rectangle_position)
                end
            end
        end
    end
  drawnow;
end